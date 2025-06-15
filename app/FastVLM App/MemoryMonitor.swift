//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import Foundation
import SwiftUI

@Observable
class MemoryMonitor {
    var currentMemoryUsage: UInt64 = 0
    var peakMemoryUsage: UInt64 = 0
    var isMonitoring = false
    
    private var monitoringTask: Task<Void, Never>?
    
    func startMonitoring() {
        guard !isMonitoring else { return }
        isMonitoring = true
        
        monitoringTask = Task {
            while !Task.isCancelled && isMonitoring {
                let usage = getMemoryUsage()
                
                await MainActor.run {
                    self.currentMemoryUsage = usage
                    if usage > self.peakMemoryUsage {
                        self.peakMemoryUsage = usage
                    }
                }
                
                // Check every 2 seconds
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }
    
    func stopMonitoring() {
        isMonitoring = false
        monitoringTask?.cancel()
        monitoringTask = nil
    }
    
    private func getMemoryUsage() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: info) / MemoryLayout<integer_t>.size)
        
        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        if kerr == KERN_SUCCESS {
            return info.resident_size
        }
        return 0
    }
    
    var memoryUsageMB: Double {
        Double(currentMemoryUsage) / 1024.0 / 1024.0
    }
    
    var peakMemoryUsageMB: Double {
        Double(peakMemoryUsage) / 1024.0 / 1024.0
    }
    
    var memoryPressureLevel: String {
        let totalMemory = ProcessInfo.processInfo.physicalMemory
        let usagePercent = Double(currentMemoryUsage) / Double(totalMemory) * 100
        
        switch usagePercent {
        case 0..<50:
            return "Low"
        case 50..<75:
            return "Medium"
        case 75..<90:
            return "High"
        default:
            return "Critical"
        }
    }
    
    deinit {
        stopMonitoring()
    }
}