// PartialScheduleTests.swift — the init-image / partial-t entry (SR probe,
// SENSENOVA-SR-EVAL §7 step 1) must leave the default path untouched.
//
// Pure functions only (no model): the schedule at startT 0 is bit-identical
// to the pre-startT form, the shift inverse round-trips, and a partial
// schedule starts where it was asked to and still ends at 1.

import Foundation
import MLX
import XCTest

@testable import SenseNovaU1

final class PartialScheduleTests: XCTestCase {

    /// The pre-startT schedule, verbatim, as the oracle.
    private func legacySchedule(numSteps: Int, shift: Float, enable: Bool) -> [Float] {
        let linear = (0 ... numSteps).map { Float($0) / Float(numSteps) }
        guard enable else { return linear }
        return linear.map { t in
            let sigma = 1 - t
            let shifted = shift * sigma / (1 + (shift - 1) * sigma)
            return 1 - shifted
        }
    }

    func testDefaultScheduleIsBitIdentical() {
        for (steps, shift, enable) in [(50, Float(3.0), true), (8, Float(3.0), true), (20, Float(1.0), true), (8, Float(3.0), false)] {
            let legacy = legacySchedule(numSteps: steps, shift: shift, enable: enable)
            let now = NEOChatModel.timestepSchedule(numSteps: steps, shift: shift, enable: enable)
            let explicit = NEOChatModel.timestepSchedule(numSteps: steps, shift: shift, enable: enable, startT: 0)
            XCTAssertEqual(legacy, now, "schedule(\(steps), \(shift), \(enable)) drifted from the legacy form")
            XCTAssertEqual(legacy, explicit, "startT: 0 must be the legacy schedule, bit for bit")
            XCTAssertEqual(legacy.map(\.bitPattern), now.map(\.bitPattern))
        }
    }

    func testShiftInverseRoundTrips() {
        for shift in [Float(1.0), 2.0, 3.0, 6.0] {
            for t in stride(from: Float(0), through: 1, by: 0.05) {
                let lin = NEOChatModel.unshiftedT(t, shift: shift, enable: true)
                let sigma = 1 - lin
                let back = 1 - shift * sigma / (1 + (shift - 1) * sigma)
                XCTAssertEqual(back, t, accuracy: 2e-6, "shift \(shift) t \(t)")
            }
            XCTAssertEqual(NEOChatModel.unshiftedT(0.4, shift: shift, enable: false), 0.4)
        }
    }

    func testPartialScheduleStartsAtStartTAndEndsAtOne() {
        for startT in [Float(0.3), 0.5, 0.65, 0.7, 0.9] {
            for steps in [1, 2, 8] {
                let s = NEOChatModel.timestepSchedule(numSteps: steps, shift: 3.0, enable: true, startT: startT)
                XCTAssertEqual(s.count, steps + 1)
                XCTAssertEqual(s[0], startT, accuracy: 2e-6)
                XCTAssertEqual(s[steps], 1, accuracy: 1e-6)
                for i in 0 ..< steps { XCTAssertLessThan(s[i], s[i + 1], "monotone") }
            }
        }
        // one step from startT is a single Euler jump of (1 − startT)
        let one = NEOChatModel.timestepSchedule(numSteps: 1, shift: 3.0, enable: true, startT: 0.7)
        XCTAssertEqual(one[1] - one[0], 0.3, accuracy: 2e-6)
    }

    func testInitialStateMixing() throws {
        try Device.withDefaultDevice(Device.cpu) {
            let noise = (MLXArray.ones([1, 3, 32, 32]) * Float(4.0)).asType(.bfloat16)
            let x = MLXArray.ones([1, 3, 32, 32]) * Float(-1.0)
            // no init image: the noise itself, untouched
            // means are taken in fp32: a bf16 reduction over 3072 elements saturates
            // (4 + 4 + … stops at 1024, and the mean reads 0.33)
            func mean32(_ a: MLXArray) -> Float { a.asType(.float32).mean().item(Float.self) }
            let plain = NEOChatModel.initialState(
                noise: noise, initImage: nil, startT: 0, mix: true, height: 32, width: 32)
            XCTAssertEqual(mean32(plain), 4.0, accuracy: 1e-6)
            // SDEdit mix at t 0.7: 0.7·(−1) + 0.3·4 = 0.5
            let mixed = NEOChatModel.initialState(
                noise: noise, initImage: x, startT: 0.7, mix: true, height: 32, width: 32)
            XCTAssertEqual(mixed.dtype, .bfloat16)
            XCTAssertEqual(mean32(mixed), 0.5, accuracy: 1e-2)
            // FluxSR-literal entry: the clean image as the state
            let clean = NEOChatModel.initialState(
                noise: noise, initImage: x, startT: 0.7, mix: false, height: 32, width: 32)
            XCTAssertEqual(mean32(clean), -1.0, accuracy: 1e-6)
        }
    }
}
