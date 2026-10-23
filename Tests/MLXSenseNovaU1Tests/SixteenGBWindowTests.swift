// SixteenGBWindowTests.swift — the 16 GB fill tier's declaration (AB-T-0278):
// `SenseNovaWindow.fill768` over the 8step-v2-4bit artifact, what MLXEngine's
// admission makes of it on a 16 GB Mac's budget, and the bounds `run()` holds
// it to. Offline (Stage-1 doctrine), except the gated live run at the bottom.

import CoreGraphics
import Foundation
import ImageIO
import MLXServeConformance
import MLXServeCore
import MLXToolKit
import XCTest

@testable import MLXSenseNovaU1

final class SixteenGBWindowTests: XCTestCase {

    static let mib: UInt64 = 1 << 20

    /// A 16 GB Mac's budget as AB-D-0141 and the governor count it: Metal's
    /// `recommendedMaxWorkingSetSize`, ≈ 74 % of 16 GiB on macOS 26+ = 12.7 GB.
    static let budget16GB: UInt64 = 12_700_000_000

    static func device16GB(_ chip: ChipTier) -> DeviceProfile {
        DeviceProfile(
            chipTier: chip, macOS: SemanticVersion(major: 27, minor: 0, patch: 1),
            backends: [.metalGPU], totalMemoryBytes: 16 << 30)
    }

    static let fill16GB = SenseNovaU1Configuration(variant: .fast4V2, window: .fill768)

    // MARK: - the declaration

    /// At or above every measured floor and 768² peak, within AB-T-0278's
    /// acceptance (resident ≤ 11.9 GB, activation ≤ 0.8 GB).
    func testDeclarationCoversTheReceipts() throws {
        let fp = try XCTUnwrap(SenseNovaWindow.fill768.footprint(for: .fast4V2))
        XCTAssertEqual(fp.quant, .int4)
        // floors: 11,230 MiB on the 16 GB M1 Pro (AB-R-0602), 11,235 on the M5 Pro and M5 Max
        XCTAssertGreaterThanOrEqual(fp.residentBytes, 11_235 * Self.mib)
        XCTAssertLessThanOrEqual(fp.residentBytes, 11_900_000_000)
        XCTAssertLessThanOrEqual(fp.peakActivationBytes, 800_000_000)
        // 768² single-reference edit peaks: 11,930 MiB (M1 Pro), 11,931 (M5 Pro, M5 Max)
        let total = fp.residentBytes + fp.peakActivationBytes
        XCTAssertGreaterThanOrEqual(total, 11_931 * Self.mib)
        XCTAssertLessThanOrEqual(total, Self.budget16GB)
        XCTAssertEqual(SenseNovaWindow.fill768.maxPixels, 768 * 768)
    }

    /// Only the measured pairing is declared; the native envelope's footprints
    /// (the 24 GB and 32 GB+ tiers') are what they were, and so is the floor.
    func testOnlyTheMeasuredPairingIsDeclared() {
        for variant in SenseNovaVariant.allCases {
            XCTAssertNil(SenseNovaWindow.native.footprint(for: variant))
            XCTAssertEqual(
                SenseNovaWindow.fill768.footprint(for: variant) != nil, variant == .fast4V2,
                "\(variant)")
            let native = SenseNovaU1Configuration(variant: variant)
            XCTAssertEqual(native.window, .native)
            XCTAssertNil(native.residentBytesHint, "\(variant)")
            XCTAssertNil(native.peakActivationBytesHint, "\(variant)")
        }
        XCTAssertEqual(Self.fill16GB.residentBytesHint, 11_800_000_000)
        XCTAssertEqual(Self.fill16GB.peakActivationBytesHint, 750_000_000)

        let requirements = SenseNovaU1Package.manifest.requirements
        XCTAssertEqual(requirements.footprints, [
            QuantFootprint(
                quant: .bf16, residentBytes: 35_500_000_000, peakActivationBytes: 4_000_000_000),
            QuantFootprint(
                quant: .int8, residentBytes: 20_500_000_000, peakActivationBytes: 4_200_000_000),
            QuantFootprint(
                quant: .int4, residentBytes: 12_200_000_000, peakActivationBytes: 4_500_000_000),
        ])
        XCTAssertEqual(requirements.chipFloor, .pro, "base chips are unmeasured (AB-A-0181)")
    }

    /// The window is the same artifact: same fetch address, MAT gate green.
    func testTheWindowFetchesTheSameArtifact() {
        XCTAssertEqual(
            Self.fill16GB.weightSources.map(\.repo),
            SenseNovaU1Configuration(variant: .fast4V2).weightSources.map(\.repo))
        let report = MaterializationConformance.check(freshConfiguration: Self.fill16GB)
        XCTAssertTrue(report.passed, "MAT gate failed for the window:\n\(report.summary)")
    }

    func testCodableKeepsTheWindowAndOlderConfigurationsDecodeNative() throws {
        let older = #"{"variant":"8step-v2-4bit","snapshotPath":""}"#
        let decoded = try JSONDecoder().decode(
            SenseNovaU1Configuration.self, from: Data(older.utf8))
        XCTAssertEqual(decoded.variant, .fast4V2)
        XCTAssertEqual(decoded.window, .native)

        let roundTrip = try JSONDecoder().decode(
            SenseNovaU1Configuration.self, from: JSONEncoder().encode(Self.fill16GB))
        XCTAssertEqual(roundTrip.variant, .fast4V2)
        XCTAssertEqual(roundTrip.window, .fill768)
    }

    // MARK: - admission (AB-T-0278 acceptance 2)

    /// The governor's own arithmetic: the configuration's split fits a 16 GB
    /// Mac's budget, counted three ways — AB-D-0141's 12.7 GB, the 12,116 MiB
    /// limit of the emulation (AB-R-0537), and ForgeCore's tripwire budget
    /// (16 GiB × 0.74). The native declaration fits none of them.
    func testGovernorCanFitTheWindowOnA16GBBudget() {
        let requirements = SenseNovaU1Package.manifest.requirements
        let native = SenseNovaU1Configuration(variant: .fast4V2)
        for budget in [Self.budget16GB, 12_116 * Self.mib, UInt64(Double(16 << 30) * 0.74)] {
            let governor = MemoryGovernor(budgetBytes: budget)
            func peak(_ c: SenseNovaU1Configuration) -> UInt64 {
                let split = governor.footprintSplit(
                    for: requirements, quant: c.quant,
                    persistentHint: c.residentBytesHint, transientHint: c.peakActivationBytesHint)
                return split.persistent + split.transient
            }
            XCTAssertEqual(peak(Self.fill16GB), 12_550_000_000)
            XCTAssertTrue(governor.canFit(peak(Self.fill16GB)), "budget \(budget)")
            XCTAssertEqual(peak(native), 16_700_000_000)
            XCTAssertFalse(governor.canFit(peak(native)), "budget \(budget)")
        }
    }

    /// The engine, as a 16 GB Pro Mac: the window is admissible (and admissible
    /// now, nothing else resident), the same artifact declared natively is not,
    /// and registration charges the window's split.
    func testEngineAdmitsTheWindowOnA16GBProMac() async throws {
        let engine = MLXServeEngine(
            device: Self.device16GB(.pro), governor: MemoryGovernor(budgetBytes: Self.budget16GB))
        let requirements = SenseNovaU1Package.manifest.requirements

        let window = await engine.admissibility(for: requirements, configuration: Self.fill16GB)
        XCTAssertTrue(window.eligibility.isEligible)
        XCTAssertTrue(window.admissible)
        XCTAssertTrue(window.admissibleNow)
        XCTAssertEqual(window.footprint, 12_550_000_000)

        let native = await engine.admissibility(
            for: requirements, configuration: SenseNovaU1Configuration(variant: .fast4V2))
        XCTAssertFalse(native.admissible)
        XCTAssertEqual(native.footprint, 16_700_000_000)

        let id = try await engine.register(
            SenseNovaU1Package.registration, configuration: Self.fill16GB,
            id: PackageID("sensenova-fill-16gb"))
        let advisory = try await engine.machineFitAdvisory(.imageEdit, package: id)
        XCTAssertEqual(advisory.projectedPeakBytes, 12_550_000_000)
    }

    /// `chipFloor` stays `.pro`: a 16 GB base-chip Mac is refused at
    /// registration, window or not, until a base-chip receipt (AB-A-0181).
    func testABaseChipIsStillRefused() async {
        let engine = MLXServeEngine(
            device: Self.device16GB(.base), governor: MemoryGovernor(budgetBytes: Self.budget16GB))
        do {
            try await engine.register(
                SenseNovaU1Package.registration, configuration: Self.fill16GB)
            XCTFail("a base chip registered SenseNova")
        } catch EngineError.ineligible(.chipBelowFloor(let required, let have)) {
            XCTAssertEqual(required, .pro)
            XCTAssertEqual(have, .base)
        } catch {
            XCTFail("wrong error for a base chip: \(error)")
        }
    }

    // MARK: - the window's bounds

    func testTheWindowRefusesWhatItWasNotMeasuredOn() {
        func check(
            _ width: Int, _ height: Int, references: Int = 1,
            guidance: Float = 1, imageGuidance: Float = 1
        ) throws {
            try SenseNovaU1Package.checkWindow(
                .fill768, width: width, height: height, references: references,
                guidance: guidance, imageGuidance: imageGuidance)
        }
        func assertOutside(_ body: @autoclosure () throws -> Void, _ line: UInt = #line) {
            XCTAssertThrowsError(try body(), line: line) { error in
                guard case SenseNovaU1PackageError.outsideWindow(.fill768, _) = error else {
                    return XCTFail("wrong error: \(error)", line: line)
                }
            }
        }

        XCTAssertNoThrow(try check(768, 768))
        XCTAssertNoThrow(try check(1024, 576), "the cap is the area, not the side")
        XCTAssertNoThrow(try check(512, 512))
        XCTAssertNoThrow(try check(768, 768, references: 0), "text-to-image")
        assertOutside(try check(1024, 1024))
        assertOutside(try check(800, 768))
        assertOutside(try check(1024, 608))
        assertOutside(try check(768, 768, references: 2))
        assertOutside(try check(768, 768, guidance: 4))
        assertOutside(try check(768, 768, imageGuidance: 2))

        XCTAssertNoThrow(
            try SenseNovaU1Package.checkWindow(
                .native, width: 2048, height: 2048, references: 3, guidance: 4,
                imageGuidance: 2),
            "native admits what it always did")
    }

    /// A window is declared only where it was measured: any other variant is
    /// refused at load, before any weights are read.
    func testAnUndeclaredWindowIsRefusedAtLoad() async {
        let package = SenseNovaU1Package(
            configuration: SenseNovaU1Configuration(variant: .fast8V2, window: .fill768))
        do {
            try await package.load()
            XCTFail("loaded a window declared for nothing")
        } catch SenseNovaU1PackageError.windowNotDeclared(let window, let variant) {
            XCTAssertEqual(window, .fill768)
            XCTAssertEqual(variant, .fast8V2)
        } catch {
            XCTFail("wrong error: \(error)")
        }
    }

    // MARK: - live: a 16 GB-profile run through the engine

    /// An engine with a 16 GB Pro Mac's profile and 12.7 GB budget admits,
    /// loads and runs a fill on the window, charging exactly the declaration;
    /// the bake-off's 1024² fixture comes in scaled to 768² (the shape
    /// AB-R-0602 measured), and a 1024² request is refused. Gated:
    /// SENSENOVA_LIVE_PKG=1 (the 8step-v2-4bit artifact under
    /// SENSENOVA_ARTIFACTS; SENSENOVA_FILL_FIXTURE overrides the fixture).
    func testLive16GBProfileFillThroughTheEngine() async throws {
        if ProcessInfo.processInfo.environment["SENSENOVA_LIVE_PKG"] != "1" {
            throw XCTSkip("live 16 GB-profile run gated behind SENSENOVA_LIVE_PKG=1")
        }
        let snapshot = LivePackageTests.artifactsRoot
            .appendingPathComponent("SenseNova-U1.5-8B-MoT-8step-v2-4bit")
        let configuration = SenseNovaU1Configuration(
            variant: .fast4V2, window: .fill768, snapshotPath: snapshot.path)
        let engine = MLXServeEngine(
            device: Self.device16GB(.pro), governor: MemoryGovernor(budgetBytes: Self.budget16GB))
        let id = try await engine.register(
            SenseNovaU1Package.registration, configuration: configuration,
            id: PackageID("sensenova-fill-16gb"))

        let inputURL = URL(fileURLWithPath:
            ProcessInfo.processInfo.environment["SENSENOVA_FILL_FIXTURE"]
            ?? "/Volumes/Satechi/Development/research/16gb-fill-kit/inputs/fill-stopsign.png")
        let input = Image(
            format: .png, data: try Data(contentsOf: inputURL), width: 1024, height: 1024)
        let prompt = "Replace the bright magenta area with a brown dog sitting on the pavement. "
            + "Match the picture's lighting, perspective and style, and keep everything else unchanged."

        let started = Date()
        let response = try await engine.run(
            IEditRequest(images: [input], prompt: prompt, seed: 7), package: id)
        let seconds = Date().timeIntervalSince(started)
        let memory = await engine.memory
        guard let edit = response as? IEditResponse else { return XCTFail("wrong response type") }
        guard let source = CGImageSourceCreateWithData(edit.image.data as CFData, nil),
            let cg = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { return XCTFail("undecodable PNG") }
        XCTAssertEqual(cg.width, 768)
        XCTAssertEqual(cg.height, 768)
        XCTAssertEqual(memory.residentBytes, 11_800_000_000)
        XCTAssertEqual(memory.transientReserveBytes, 750_000_000)
        XCTAssertLessThanOrEqual(memory.residentBytes + memory.transientReserveBytes, memory.budgetBytes)
        print(String(
            format: "  [live] 16 GB profile: 768² fill in %.1f s (cold), charged %.2f + %.2f GB "
                + "of %.2f GB, phys_footprint %.2f GB", seconds,
            Double(memory.residentBytes) / 1e9, Double(memory.transientReserveBytes) / 1e9,
            Double(memory.budgetBytes) / 1e9, Double(memory.realResidentBytes ?? 0) / 1e9))
        try? edit.image.data.write(
            to: LivePackageTests.artifactsRoot.appendingPathComponent("live_fill16gb_smoke.png"))

        do {
            _ = try await engine.run(
                IEditRequest(images: [input], prompt: prompt, width: 1024, height: 1024, seed: 7),
                package: id)
            XCTFail("the window ran a 1024² fill")
        } catch SenseNovaU1PackageError.outsideWindow(.fill768, _) {
        } catch {
            XCTFail("wrong error for a 1024² fill: \(error)")
        }
        await engine.evict(package: id)
    }
}
