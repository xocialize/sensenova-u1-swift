// LiveSurfaceTests.swift — live gates for the three surfaces the demo app drives,
// including the seams that only exist for the app: run-phase reporting, the
// image-guidance axis, and VQA. Gated behind SENSENOVA_LIVE_PKG=1.

import Foundation
import MLXToolKit
import XCTest

@testable import MLXSenseNovaU1

final class LiveSurfaceTests: XCTestCase {

    func requireLive() throws {
        if ProcessInfo.processInfo.environment["SENSENOVA_LIVE_PKG"] != "1" {
            throw XCTSkip("live surface gates need SENSENOVA_LIVE_PKG=1")
        }
    }

    func basePackage() -> SenseNovaU1Package {
        let snapshot = LivePackageTests.artifactsRoot
            .appendingPathComponent("SenseNova-U1.5-8B-MoT-8bit")
        return SenseNovaU1Package(
            configuration: .init(variant: .q8, snapshotPath: snapshot.path))
    }

    /// The 8-step V2 distill tier, int8 — the tier whose edit and VQA gates
    /// were lifted on evidence (AB-R-0490).
    func distillPackage() -> SenseNovaU1Package {
        let snapshot = LivePackageTests.artifactsRoot
            .appendingPathComponent("SenseNova-U1.5-8B-MoT-8step-v2-8bit")
        return SenseNovaU1Package(
            configuration: .init(variant: .fast8V2, snapshotPath: snapshot.path))
    }

    func oracleRender() throws -> Image {
        // SENSENOVA_ORACLE_RENDER overrides the sibling-checkout path (the oracle
        // tree is not a sibling when the package lives under PROD/).
        let inputURL = ProcessInfo.processInfo.environment["SENSENOVA_ORACLE_RENDER"]
            .map { URL(fileURLWithPath: $0) }
            ?? URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("sensenova-u1-oracle/renders/sn_1024_s20.png")
        return Image(format: .png, data: try Data(contentsOf: inputURL), width: 1024, height: 1024)
    }

    /// Distill tiers serve imageEdit: with no steps/guidance given the run
    /// takes the tier's defaults (8 steps, cfg 1 — the vendor's 8-step edit
    /// recipe) and returns a decodable PNG at the requested size.
    func testDistillTierEditsAtTierDefaults() async throws {
        try requireLive()
        let package = distillPackage()
        try await package.load()
        let response: any CapabilityResponse
        do {
            response = try await package.run(
                IEditRequest(
                    images: [try oracleRender()],
                    prompt: "Make the chair emerald green velvet. Preserve everything else.",
                    width: 512, height: 512, seed: 42))
        } catch {
            await package.unload()
            throw error
        }
        await package.unload()
        guard let edit = response as? IEditResponse else { return XCTFail("wrong response type") }
        XCTAssertGreaterThan(edit.image.data.count, 10_000)
        let size = LivePackageTests().decodePNGSize(edit.image.data)
        XCTAssertEqual(size?.0, 512)
        XCTAssertEqual(size?.1, 512)
        try? edit.image.data.write(
            to: LivePackageTests.artifactsRoot.appendingPathComponent("live_edit_distill_smoke.png"))
        print("  [live] edit fast8V2 (tier defaults): \(edit.image.data.count) bytes PNG @ \(size!)")
    }

    /// Distill tiers serve VQA, and the understanding stream is the base
    /// tier's: the answer must be grounded exactly as the base tier's is.
    func testDistillTierAnswersVQA() async throws {
        try requireLive()
        let package = distillPackage()
        try await package.load()
        let text: String
        do {
            let response = try await package.run(
                ImageAnalysisRequest(
                    image: try oracleRender(),
                    prompt: "What animal is shown, and what colour is the chair? Answer briefly.",
                    metaData: ["maxTokens": .int(80)]))
            text = (response as! ImageAnalysisResponse).text
        } catch {
            await package.unload()
            throw error
        }
        await package.unload()
        let lower = text.lowercased()
        XCTAssertTrue(lower.contains("cat"), "answer not grounded in the image: \(text)")
        XCTAssertTrue(lower.contains("red"), "answer missed the chair colour: \(text)")
        print("  [live] vqa fast8V2: \(text.prefix(160))")
    }

    /// P1: a consumer bound to the engine's RunMonitor must see monotonic
    /// denoise steps — without this the demo shows a dead spinner for minutes.
    func testDenoiseProgressIsReported() async throws {
        try requireLive()
        let package = basePackage()
        try await package.load()

        // The engine binds this sink around run(); here we bind it directly to
        // prove the PACKAGE emits (the engine-side plumbing is engine-tested).
        final class Box: @unchecked Sendable { var reports: [RunPhaseReport] = [] }
        let box = Box()
        do {
            _ = try await RunProgress.$sink.withValue({ box.reports.append($0) }) {
                try await package.run(
                    T2IRequest(prompt: "a red apple on a table", width: 512, height: 512,
                               steps: 4, seed: 1))
            }
        } catch {
            await package.unload()
            throw error
        }
        await package.unload()

        let denoise = box.reports.filter { $0.phase == .denoise }
        XCTAssertEqual(denoise.count, 4, "expected one report per denoise step")
        XCTAssertEqual(denoise.map(\.step), [1, 2, 3, 4], "steps must be monotonic from 1")
        XCTAssertEqual(denoise.first?.totalSteps, 4, "totalSteps must be carried")
        print("  [live] denoise reports: \(denoise.count)")
    }

    /// P2: the image-guidance axis must actually change the output (a slider that
    /// does nothing is worse than no slider).
    func testImageGuidanceChangesTheResult() async throws {
        try requireLive()
        let package = basePackage()
        try await package.load()

        // SENSENOVA_ORACLE_RENDER overrides the sibling-checkout path (the oracle
        // tree is not a sibling when the package lives under PROD/).
        let inputURL = ProcessInfo.processInfo.environment["SENSENOVA_ORACLE_RENDER"]
            .map { URL(fileURLWithPath: $0) }
            ?? URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("sensenova-u1-oracle/renders/sn_1024_s20.png")
        let image = Image(
            format: .png, data: try Data(contentsOf: inputURL), width: 1024, height: 1024)

        func edit(_ guidance: Double?) async throws -> Data {
            var meta: MetaData = [:]
            if let guidance { meta["imageGuidance"] = .double(guidance) }
            let response = try await package.run(
                IEditRequest(
                    images: [image], prompt: "Make the chair bright yellow.",
                    width: 512, height: 512, steps: 4, seed: 7, metaData: meta))
            return (response as! IEditResponse).image.data
        }

        do {
            let baseline = try await edit(nil)          // imgCfg 1.0 — two branches
            let guided = try await edit(2.5)            // imgCfg 2.5 — three branches
            XCTAssertNotEqual(
                baseline, guided,
                "imageGuidance had no effect — the uncond branch is not wired")
            print("  [live] imageGuidance 1.0 vs 2.5: \(baseline.count) vs \(guided.count) bytes")

            // and it must reject values the contract does not allow
            do {
                _ = try await edit(9.0)
                XCTFail("out-of-range imageGuidance was accepted")
            } catch is SenseNovaU1PackageError {
                // expected
            }
        } catch {
            await package.unload()
            throw error
        }
        await package.unload()
    }

    /// P3: VQA on the same resident model, through the canonical contract.
    func testImageAnalysisAnswers() async throws {
        try requireLive()
        let package = basePackage()
        try await package.load()

        // SENSENOVA_ORACLE_RENDER overrides the sibling-checkout path (the oracle
        // tree is not a sibling when the package lives under PROD/).
        let inputURL = ProcessInfo.processInfo.environment["SENSENOVA_ORACLE_RENDER"]
            .map { URL(fileURLWithPath: $0) }
            ?? URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("sensenova-u1-oracle/renders/sn_1024_s20.png")
        let image = Image(
            format: .png, data: try Data(contentsOf: inputURL), width: 1024, height: 1024)

        let text: String
        do {
            let response = try await package.run(
                ImageAnalysisRequest(
                    image: image,
                    prompt: "What animal is shown, and what colour is the chair? Answer briefly.",
                    metaData: ["maxTokens": .int(80)]))
            text = (response as! ImageAnalysisResponse).text
        } catch {
            await package.unload()
            throw error
        }
        await package.unload()

        XCTAssertFalse(text.isEmpty, "VQA returned nothing")
        let lower = text.lowercased()
        XCTAssertTrue(lower.contains("cat"), "answer not grounded in the image: \(text)")
        XCTAssertTrue(lower.contains("red"), "answer missed the chair colour: \(text)")
        print("  [live] vqa: \(text.prefix(160))")
    }

    // MARK: - TailSteer tier

    /// The TailSteer multi-turn tier through the package: load by variant, edit
    /// at the tier defaults, and answer VQA on its own understanding stream.
    /// Gated on the artifact being present (SENSENOVA_TAILSTEER_SNAPSHOT, or the
    /// artifacts root).
    func testTailSteerTierEditsAndAnswersVQA() async throws {
        try requireLive()
        let path = ProcessInfo.processInfo.environment["SENSENOVA_TAILSTEER_SNAPSHOT"]
            ?? LivePackageTests.artifactsRoot
                .appendingPathComponent("SenseNova-U1.5-8B-MoT-TailSteer-8step-v2-8bit").path
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("no TailSteer artifact at \(path)")
        }
        let package = SenseNovaU1Package(
            configuration: .init(variant: .fast8V2TailSteer, snapshotPath: path))
        try await package.load()
        let edit: IEditResponse
        let text: String
        do {
            edit = try await package.run(
                IEditRequest(
                    images: [try oracleRender()],
                    prompt: "Make the chair emerald green velvet. Preserve everything else.",
                    width: 512, height: 512, seed: 42)) as! IEditResponse
            text = (try await package.run(
                ImageAnalysisRequest(
                    image: try oracleRender(),
                    prompt: "What animal is shown, and what colour is the chair? Answer briefly.",
                    metaData: ["maxTokens": .int(80)])) as! ImageAnalysisResponse).text
        } catch {
            await package.unload()
            throw error
        }
        await package.unload()
        XCTAssertGreaterThan(edit.image.data.count, 10_000)
        let size = LivePackageTests().decodePNGSize(edit.image.data)
        XCTAssertEqual(size?.0, 512)
        XCTAssertEqual(size?.1, 512)
        let lower = text.lowercased()
        XCTAssertTrue(lower.contains("cat"), "answer not grounded in the image: \(text)")
        XCTAssertTrue(lower.contains("red"), "answer missed the chair colour: \(text)")
        try? edit.image.data.write(
            to: URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("live_edit_tailsteer_smoke.png"))
        print("  [live] tailsteer edit \(edit.image.data.count) bytes @ \(size!) · vqa: \(text.prefix(160))")
    }
}
