//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import SwiftUI
import MLX

struct DebugView: View {
    let modelManager: ModelManager
    @State private var debugInfo: [String] = []
    @State private var autoRefresh = true
    
    var body: some View {
        NavigationView {
            List {
                Section("Primary Model Information") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Selected Model: \(modelManager.selectedModelType.rawValue)")
                        Text("Model Info: \(modelManager.currentModel.modelInfo)")
                        Text("Running: \(modelManager.running ? "Yes" : "No")")
                        Text("Output Length: \(modelManager.output.count) characters")
                        if !modelManager.promptTime.isEmpty {
                            Text("Last Processing Time: \(modelManager.promptTime)")
                        }
                    }
                    .font(.caption)
                }
                
                Section("Secondary Model Information") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Secondary Model: \(modelManager.secondaryModelInfo)")
                        Text("All Models Status: \(modelManager.allModelsStatus)")
                    }
                    .font(.caption)
                }
                
                Section("MLX Memory Information") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("GPU Cache Limit: \(MLX.GPU.cacheLimit / 1024 / 1024) MB")
                        Text("GPU Memory Limit: \(MLX.GPU.memoryLimit / 1024 / 1024) MB")
                        Text("GPU Peak Memory: \(MLX.GPU.peakMemory / 1024 / 1024) MB")
                    }
                    .font(.caption)
                }
                
                Section("System Information") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Physical Memory: \(ProcessInfo.processInfo.physicalMemory / 1024 / 1024) MB")
                        Text("Active Processor Count: \(ProcessInfo.processInfo.activeProcessorCount)")
                        #if os(iOS)
                        Text("Available Memory: \(os_proc_available_memory() / 1024 / 1024) MB")
                        #endif
                    }
                    .font(.caption)
                }
                
                Section("Debug Output") {
                    ForEach(debugInfo.indices, id: \.self) { index in
                        Text(debugInfo[index])
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .navigationTitle("Debug Information")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("Clear Debug") {
                        debugInfo.removeAll()
                    }
                }
                
                ToolbarItem(placement: .secondaryAction) {
                    Button("Refresh") {
                        refreshDebugInfo()
                    }
                }
                
                ToolbarItem(placement: .secondaryAction) {
                    Button(autoRefresh ? "Stop Auto-Refresh" : "Start Auto-Refresh") {
                        autoRefresh.toggle()
                    }
                }
            }
            .onAppear {
                refreshDebugInfo()
            }
            .onReceive(Timer.publish(every: 2.0, on: .main, in: .common).autoconnect()) { _ in
                if autoRefresh {
                    refreshDebugInfo()
                }
            }
        }
    }
    
    private func refreshDebugInfo() {
        let timestamp = Date().formatted(.dateTime.hour().minute().second())
        debugInfo.append("Refreshed at \(timestamp)")
        debugInfo.append("Current Model: \(modelManager.selectedModelType.rawValue)")
        debugInfo.append("Models Status: \(modelManager.allModelsStatus)")
        
        // Keep only last 50 debug entries
        if debugInfo.count > 50 {
            debugInfo.removeFirst(debugInfo.count - 50)
        }
    }
}

#Preview {
    DebugView(modelManager: ModelManager())
}
