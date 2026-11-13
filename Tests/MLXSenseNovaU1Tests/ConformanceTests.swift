// ConformanceTests.swift — offline conformance for the MLXSenseNovaU1 wrapper:
// manifest sanity, the fetch-vs-origin provenance split, the MAT gate
// (MaterializationConformance) per variant, and the CAN gate
// (CancellationConformance: pre-cancelled propagation + cadence declaration).
// Nothing here touches weights or runs a kernel (Stage-1 doctrine).

import Foundation
import MLXServeConformance
import MLXToolKit
import SenseNovaU1
import XCTest

@testable import MLXSenseNovaU1

final class ConformanceTests: XCTestCase {

    func testManifestSanity() {
        let m = SenseNovaU1Package.manifest
        XCTAssertEqual(m.license.weightLicense, .apache2)
        XCTAssertEqual(m.license.portCodeLicense, .mit)
        XCTAssertEqual(m.provenance.tier, 3)

        // every variant's quant tier carries a split footprint
        let quants = Set(m.requirements.footprints.map(\.quant))
        for variant in SenseNovaVariant.allCases {
            XCTAssertTrue(quants.contains(variant.quant), "no footprint for \(variant)")
        }
        for fp in m.requirements.footprints {
            XCTAssertGreaterThan(fp.residentBytes, 0)
            XCTAssertGreaterThan(fp.peakActivationBytes, 0, "flat footprint (activation folded in?)")
        }

        let caps = Set(m.surfaces.map(\.capability))
        XCTAssertEqual(caps, [.textToImage, .imageEdit, .imageAnalysis])
    }

    /// Every tier serves every declared surface — including the 8-step distill
    /// tiers, whose edit and VQA refusals were lifted on evidence (AB-R-0490):
    /// the adapter touches only gen-stream projections, VQA is token-identical
    /// to the base tier, and the 8-step edit matches the 50-step base edit.
    /// `run()` consults the same declaration, so this pins the policy.
    func testEveryTierServesEverySurface() {
        let declared = Set(SenseNovaU1Package.manifest.surfaces.map(\.capability))
        XCTAssertEqual(declared, [.textToImage, .imageEdit, .imageAnalysis])
        for variant in SenseNovaVariant.allCases {
            XCTAssertEqual(variant.capabilities, declared, "\(variant) must serve every surface")
        }
        for variant in SenseNovaVariant.allCases where variant.isDistilled {
            XCTAssertTrue(variant.capabilities.contains(.imageEdit), "\(variant) edit")
            XCTAssertTrue(variant.capabilities.contains(.imageAnalysis), "\(variant) VQA")
            XCTAssertEqual(variant.defaultSteps, 8)
            XCTAssertEqual(variant.defaultGuidance, 1.0)
        }
    }

    /// The split-provenance rule: fetch address = OUR namespace, origin = theirs,
    /// and the trailing model name survives the mirror.
    func testProvenanceSplit() {
        let m = SenseNovaU1Package.manifest
        XCTAssertTrue(m.provenance.sourceRepo.hasPrefix("sensenova/"), "origin must be upstream")
        for variant in SenseNovaVariant.allCases {
            let config = SenseNovaU1Configuration(variant: variant)
            for source in config.weightSources {
                XCTAssertTrue(
                    source.repo.hasPrefix("mlx-community/"),
                    "\(variant): fetch address must be ours (\(source.repo))")
                XCTAssertTrue(
                    source.repo.contains("SenseNova-U1.5-8B-MoT"),
                    "\(variant): artifact name must carry the upstream checkpoint name")
            }
        }
    }

    /// The TailSteer tier is a distill tier over its own upstream checkpoint: the
    /// fetch address is ours, the origin is the TailSteer repo, and the sampler
    /// contract is the 8-step one.
    func testTailSteerTier() {
        let v = SenseNovaVariant.fast8V2TailSteer
        XCTAssertEqual(v.repo, "mlx-community/SenseNova-U1.5-8B-MoT-TailSteer-8step-v2-8bit")
        XCTAssertEqual(v.quant, .int8)
        XCTAssertTrue(v.isDistilled)
        XCTAssertFalse(v.isPoseTuned)
        XCTAssertEqual(v.upstream.repo, "sensenova/SenseNova-U1.5-8B-MoT-TailSteer")
        for other in SenseNovaVariant.allCases where other != v {
            XCTAssertEqual(other.upstream.repo, SenseNovaU1Package.manifest.provenance.sourceRepo)
            XCTAssertEqual(other.upstream.revision, SenseNovaU1Package.manifest.provenance.revision)
        }
    }

    func testMATGatePerVariant() {
        for variant in SenseNovaVariant.allCases {
            let report = MaterializationConformance.check(
                freshConfiguration: SenseNovaU1Configuration(variant: variant))
            XCTAssertTrue(report.passed, "MAT gate failed for \(variant):\n\(report.summary)")
        }
    }

    /// CAN-1/2: a pre-cancelled run must propagate CancellationError unchanged
    /// (the entry checkpoint precedes notLoaded validation, so no weights needed).
    func testCANPreCancelledRun() async {
        let package = SenseNovaU1Package(configuration: SenseNovaU1Configuration())
        let report = await CancellationConformance.checkRun(
            package: package, request: T2IRequest(prompt: "conformance probe"))
        XCTAssertTrue(report.passed, "CAN run gate failed:\n\(report.summary)")
    }

    // MARK: - Pose specialty (C6)

    /// The pose tiers are a base-checkpoint merge, not a distill: every guard
    /// that turns a surface off for the 8-step tiers must stay open for them.
    func testPoseTiersKeepEverySurface() {
        for variant in SenseNovaVariant.allCases where variant.isPoseTuned {
            XCTAssertFalse(variant.isDistilled, "\(variant) must not be treated as a distill tier")
            XCTAssertEqual(variant.defaultGuidance, 4.0, "\(variant) needs true CFG")
            XCTAssertEqual(variant.defaultSteps, 28, "\(variant): 28 is the receipted setting")
        }
        XCTAssertEqual(SenseNovaVariant.pose8.quant, .int8)
        XCTAssertEqual(SenseNovaVariant.poseBf16.quant, .bf16)
    }

    /// C6: the specialty is declared, uses the REGISTERED vocabulary, and is
    /// scoped to the pose tiers — advertising it package-wide would tell the
    /// Model Manager the plain tiers apply a skeleton, which they do not.
    func testPoseSpecialtyIsRegisteredAndScoped() {
        let pose = SenseNovaU1Package.poseManifest
        XCTAssertFalse(pose.specialties.isEmpty, "pose manifest declares no specialty")
        for weight in pose.specialties {
            XCTAssertTrue(
                weight.specialty.isRegistered,
                "unregistered specialty \(weight.specialty.rawValue) — C6 vocabulary drift")
        }
        XCTAssertEqual(pose.specialties.map(\.specialty), [.poseDriven])
        XCTAssertEqual(Set(pose.capabilities), [.imageEdit],
                       "the pose manifest must not hijack the other capabilities' default routing")
        XCTAssertTrue(SenseNovaU1Package.manifest.specialties.isEmpty,
                      "the shared manifest must not advertise pose for the plain tiers")

        for variant in SenseNovaVariant.allCases {
            XCTAssertEqual(
                variant.specialties.isEmpty, !variant.isPoseTuned,
                "\(variant) advertises the wrong specialty set")
        }
    }

    /// The pose registration must refuse a configuration whose weights carry no
    /// pose adapter — otherwise `.poseDriven` would rank tiers that cannot do it.
    func testPoseRegistrationRefusesNonPoseVariant() throws {
        let registration = SenseNovaU1Package.poseRegistration
        XCTAssertThrowsError(
            try registration.makePackage(SenseNovaU1Configuration(variant: .q8)),
            "the pose registration accepted a plain tier"
        ) { error in
            guard case PackageError.configurationMismatch = error else {
                return XCTFail("wrong error for a non-pose variant: \(error)")
            }
        }
        XCTAssertNoThrow(
            try registration.makePackage(SenseNovaU1Configuration(variant: .pose8)))
        XCTAssertNoThrow(
            try registration.makePackage(SenseNovaU1Configuration(variant: .poseBf16)))
    }

    /// The two-slot prompt is the contract with the adapter: two placeholders,
    /// pose first. It has to survive the placeholder expander with exactly two
    /// images — a mismatch there throws rather than silently dropping a slot.
    func testPosePromptExpandsToTwoImageSlots() throws {
        let prompt = SenseNovaVariant.posePrompt
        XCTAssertEqual(prompt.components(separatedBy: "<image>").count - 1, 2)
        let poseSlot = try XCTUnwrap(prompt.range(of: "Image-1"))
        let refSlot = try XCTUnwrap(prompt.range(of: "Image-2"))
        XCTAssertLessThan(poseSlot.lowerBound, refSlot.lowerBound, "slot order is load-bearing")

        let expanded = try Conversation.expandImagePlaceholders(
            prompt: prompt, imageTokenCounts: [576, 576])
        XCTAssertFalse(expanded.contains("<image>"), "a placeholder survived expansion")
        XCTAssertEqual(
            expanded.components(separatedBy: Conversation.imgContextToken).count - 1, 1152)
        // and it must still reject a slot count the images cannot back
        XCTAssertThrowsError(
            try Conversation.expandImagePlaceholders(prompt: prompt, imageTokenCounts: [576]))
    }

    /// CAN-3: long-run manifests declare their checkpoint cadence.
    func testCANCadenceDeclaration() {
        let report = CancellationConformance.checkCadence(
            manifest: SenseNovaU1Package.manifest,
            posture: .cadence([
                CancellationConformance.CheckpointCadence(phase: "denoise", unit: .step),
                CancellationConformance.CheckpointCadence(phase: "decode", unit: .token),
            ]))
        XCTAssertTrue(report.passed, "CAN cadence gate failed:\n\(report.summary)")
    }
}
