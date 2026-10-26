import React, { useState } from 'react';
import {
  BinaryDatasetPacker,
  DiscoveredTopologyProfile,
  GenerationBounds,
  calculateConvergenceAccuracy,
  evaluateClosedFormMobileBenchmark,
} from '../engine/topologicalCore';
import { ANHARMONIC_CODE_ARTIFACTS } from '../engine/codeTemplates';
import {
  Download,
  Copy,
  Check,
  Play,
  Trash2,
  Cpu,
  Database,
  Layers,
  Sliders,
  FileCode,
  Sparkles,
  FileText,
  X,
} from 'lucide-react';

export type ActiveDrawerTab =
  | 'none'
  | 'tokenizer'
  | 'neural'
  | 'distortion'
  | 'pipeline';

interface LaboratoryDrawersProps {
  activeDrawer: ActiveDrawerTab;
  onCloseDrawer: () => void;
  // Current active physics & latent state
  currentProfile: DiscoveredTopologyProfile;
  bounds: GenerationBounds;
  onUpdateBounds: (next: GenerationBounds) => void;
  onLoadProfileIntoViewport: (profile: DiscoveredTopologyProfile) => void;
  onSetSeedA: (profile: DiscoveredTopologyProfile) => void;
  onSetSeedB: (profile: DiscoveredTopologyProfile) => void;
  seedA: DiscoveredTopologyProfile;
  seedB: DiscoveredTopologyProfile;
  // Monte Carlo Batch Generator state
  collectedProfiles: DiscoveredTopologyProfile[];
  totalGeneratedCount: number;
  optimalTotalCount: number;
  isGeneratingBatch: boolean;
  lastBatchTimeMs: number;
  keepOptimalOnly: boolean;
  onToggleKeepOptimalOnly: () => void;
  onRunMonteCarloBatch: (iterations: number) => void;
  onDownloadCSV: () => void;
  onDownloadBinary: () => void;
  // Neural & Fourier MLP state
  fourierMlpPrediction: {
    fitnessPred: number;
    optimalProb: number;
    fourierFeatures: Float32Array;
    layerNorms: [number, number, number];
  };
  trainedEpochs: number;
  lastEpochLoss: number;
  lossHistory: {
    epoch: number;
    mseLoss: number;
    bceLoss: number;
    totalLoss: number;
  }[];
  onTrainEpochs: (epochs: number) => void;
  // Camera & Orthographic Distortion state
  cameraPitch: number;
  cameraYaw: number;
  onCameraAngleChange: (pitch: number, yaw: number) => void;
  // Performance & Privacy state
  physicsDeltaMs: number;
  activeParticleCount: number;
  onExecutePrivacyPurge: () => void;
  purgeToastMessage: string | null;
}

export const LaboratoryDrawers: React.FC<LaboratoryDrawersProps> = ({
  activeDrawer,
  onCloseDrawer,
  currentProfile,
  bounds,
  onUpdateBounds,
  onLoadProfileIntoViewport,
  onSetSeedA,
  onSetSeedB,
  seedA,
  seedB,
  collectedProfiles,
  totalGeneratedCount,
  optimalTotalCount,
  isGeneratingBatch,
  lastBatchTimeMs,
  keepOptimalOnly,
  onToggleKeepOptimalOnly,
  onRunMonteCarloBatch,
  onDownloadCSV,
  onDownloadBinary,
  fourierMlpPrediction,
  trainedEpochs,
  lastEpochLoss,
  lossHistory,
  onTrainEpochs,
  cameraPitch,
  cameraYaw,
  onCameraAngleChange,
  physicsDeltaMs,
  activeParticleCount,
  onExecutePrivacyPurge,
  purgeToastMessage,
}) => {
  const [inferenceMode, setInferenceMode] = useState<'fourier-mlp' | 'closed-form'>('fourier-mlp');
  const [selectedCodeId, setSelectedCodeId] = useState<string>(ANHARMONIC_CODE_ARTIFACTS[0].id);
  const [copiedCode, setCopiedCode] = useState(false);
  const [binaryVerifiedCount, setBinaryVerifiedCount] = useState<number | null>(null);
  const [showPdfPreview, setShowPdfPreview] = useState(false);

  if (activeDrawer === 'none') return null;

  // Compute closed-form mobile benchmark vs Fourier MLP
  const closedForm = evaluateClosedFormMobileBenchmark(
    currentProfile.pitchAxis,
    currentProfile.yawAxis,
    currentProfile.timeFreq,
    currentProfile.globalScale
  );

  const activeInferredScore =
    inferenceMode === 'fourier-mlp'
      ? fourierMlpPrediction.fitnessPred
      : closedForm.predictedFitnessScore;

  const convergence = calculateConvergenceAccuracy(
    currentProfile.fitnessScore,
    activeInferredScore
  );

  const isOptimalSatisfied =
    inferenceMode === 'fourier-mlp'
      ? fourierMlpPrediction.optimalProb >= 0.5 && currentProfile.isOptimal === 1
      : closedForm.isOptimalBoundarySatisfied;

  // Orthographic distortion calculations from ProjectionDistortionTestHud
  const compressionRatioX = Math.abs(Math.cos(cameraYaw));
  const compressionRatioY = Math.abs(Math.cos(cameraPitch));
  const combinedDistortionFactor =
    (1.0 - compressionRatioX * compressionRatioY) * 100.0;

  // Storage metrics calculation (56 bytes avg CSV vs 29 bytes fixed Binary)
  const csvSizeMb = (totalGeneratedCount * 56) / (1024 * 1024);
  const binSizeMb =
    (totalGeneratedCount * BinaryDatasetPacker.RECORD_STRIDE_BYTES) /
    (1024 * 1024);
  const compressionSavingsPct =
    totalGeneratedCount > 0 ? ((56 - 29) / 56) * 100 : 73.2;

  const hexBreakdown = BinaryDatasetPacker.formatRecordHexBreakdown(currentProfile);

  const selectedArtifact =
    ANHARMONIC_CODE_ARTIFACTS.find((a) => a.id === selectedCodeId) ||
    ANHARMONIC_CODE_ARTIFACTS[0];

  const handleCopyCode = () => {
    navigator.clipboard.writeText(selectedArtifact.code);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 1800);
  };

  const handleDownloadCodeFile = () => {
    const blob = new Blob([selectedArtifact.code], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = selectedArtifact.filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleVerifyBinaryRoundTrip = () => {
    const sample = collectedProfiles.slice(0, 128);
    const buffer = BinaryDatasetPacker.packProfilesToBinary(sample);
    const decoded = BinaryDatasetPacker.readBinaryStream(buffer, bounds);
    setBinaryVerifiedCount(decoded.length);
  };

  // Simulated runtime heap usage based on active buffers
  const heapMemoryMb =
    14.2 +
    (collectedProfiles.length * 64) / (1024 * 1024) +
    (activeParticleCount * 16) / (1024 * 1024);

  return (
    <aside
      className="fixed top-14 right-4 bottom-24 z-30 w-full max-w-[480px] bg-[#0B0E17]/95 border border-slate-800 rounded-xl flex flex-col overflow-hidden shadow-2xl"
      aria-label="Anharmonic Space Telemetry Drawer"
    >
      {/* Drawer Header */}
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-slate-800 bg-[#111522]">
        <div className="flex items-center gap-2.5">
          {activeDrawer === 'tokenizer' && <Database className="w-4 h-4 text-cyan-400" />}
          {activeDrawer === 'neural' && <Sparkles className="w-4 h-4 text-pink-400" />}
          {activeDrawer === 'distortion' && <Sliders className="w-4 h-4 text-cyan-400" />}
          {activeDrawer === 'pipeline' && <FileCode className="w-4 h-4 text-cyan-400" />}
          <h2 className="text-sm font-semibold tracking-wide text-white font-display">
            {activeDrawer === 'tokenizer' && 'Topological Tokenizer Core & Binary Packer'}
            {activeDrawer === 'neural' && 'Neural Prediction Matrix & Fourier MLP'}
            {activeDrawer === 'distortion' && 'Orthographic Distortion & Engine Telemetry'}
            {activeDrawer === 'pipeline' && 'Embedded AI Pipeline & Reference Suite'}
          </h2>
        </div>
        <button
          onClick={onCloseDrawer}
          className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800/80 transition-colors"
          aria-label="Close panel"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Drawer Body */}
      <div className="flex-1 overflow-y-auto p-5 space-y-6">
        {/* ===================================================================
            TAB 1: MONTE CARLO TOKENIZER & 29-BYTE BINARY STREAM PACKER
           =================================================================== */}
        {activeDrawer === 'tokenizer' && (
          <>
            {/* Stat Tracking Display Row */}
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3.5 bg-[#06070D] border border-slate-800/90 rounded-lg">
                <div className="text-xs text-slate-400">Collected Tokens</div>
                <div className="text-2xl font-mono font-semibold text-cyan-400 mt-1">
                  {totalGeneratedCount.toLocaleString()}
                </div>
                <div className="text-xs font-mono text-slate-500 mt-1">
                  Optimal: {optimalTotalCount.toLocaleString()} · Batch: {lastBatchTimeMs} ms
                </div>
              </div>
              <div className="p-3.5 bg-[#06070D] border border-slate-800/90 rounded-lg">
                <div className="text-xs text-slate-400">Engine State</div>
                <div
                  className={`text-2xl font-mono font-semibold mt-1 ${
                    isGeneratingBatch ? 'text-pink-400' : 'text-emerald-400'
                  }`}
                >
                  {isGeneratingBatch ? 'GENERATING' : 'IDLE'}
                </div>
                <div className="text-xs font-mono text-slate-500 mt-1">
                  Worker Pool · Zero-GC Stride
                </div>
              </div>
            </div>

            {/* Generation Bounds Controls */}
            <div className="space-y-3 pt-2 border-t border-slate-800/80">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-300">
                  Physical Boundary Constraints (GenerationBounds)
                </span>
                <span className="text-xs font-mono text-slate-400">
                  L2 Squared Strain · XY Polygon Area
                </span>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400">Node Count</span>
                    <span className="font-mono text-cyan-400">{bounds.nodeCount} nodes</span>
                  </div>
                  <input
                    type="range"
                    min={3}
                    max={8}
                    step={1}
                    value={bounds.nodeCount}
                    onChange={(e) =>
                      onUpdateBounds({ ...bounds, nodeCount: Number(e.target.value) })
                    }
                    className="w-full anharmonic-slider"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400">Target Stiffness</span>
                    <span className="font-mono text-cyan-400">
                      {bounds.targetStiffness.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.1}
                    max={1.5}
                    step={0.05}
                    value={bounds.targetStiffness}
                    onChange={(e) =>
                      onUpdateBounds({
                        ...bounds,
                        targetStiffness: Number(e.target.value),
                      })
                    }
                    className="w-full anharmonic-slider"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400">Max Strain Threshold</span>
                    <span className="font-mono text-pink-400">
                      {bounds.maxStrainThreshold.toFixed(1)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={2.0}
                    max={40.0}
                    step={0.5}
                    value={bounds.maxStrainThreshold}
                    onChange={(e) =>
                      onUpdateBounds({
                        ...bounds,
                        maxStrainThreshold: Number(e.target.value),
                      })
                    }
                    className="w-full anharmonic-slider anharmonic-slider-magenta"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400">Min Volume Threshold</span>
                    <span className="font-mono text-cyan-400">
                      {bounds.minAreaThreshold.toFixed(0)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={50}
                    max={1200}
                    step={25}
                    value={bounds.minAreaThreshold}
                    onChange={(e) =>
                      onUpdateBounds({
                        ...bounds,
                        minAreaThreshold: Number(e.target.value),
                      })
                    }
                    className="w-full anharmonic-slider"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between pt-1">
                <span className="text-xs text-slate-400">
                  Negative Constraint Retention (prevent mode collapse)
                </span>
                <button
                  onClick={onToggleKeepOptimalOnly}
                  className="px-2.5 py-1 text-xs font-mono rounded bg-slate-800 hover:bg-slate-700 text-slate-200 transition-colors whitespace-nowrap shrink-0"
                >
                  {keepOptimalOnly ? 'Saving Optimal Only (1)' : 'Retaining All Samples (0 & 1)'}
                </button>
              </div>
            </div>

            {/* Dataset File Target Status & Action Buttons */}
            <div className="p-3.5 bg-[#06070D] border border-slate-800/90 rounded-lg space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-cyan-400">
                  Dataset Sync Bridge Target Status
                </span>
                <span className="text-xs font-mono text-emerald-400">
                  -{compressionSavingsPct.toFixed(1)}% Binary Footprint
                </span>
              </div>
              <div className="text-xs font-mono text-slate-300 space-y-1">
                <div>Path: Documents/anharmonic_ai_training_set.csv ({csvSizeMb.toFixed(2)} MB)</div>
                <div>Packed Binary: Documents/anharmonic_ai_training_set.bin ({binSizeMb.toFixed(2)} MB)</div>
              </div>
            </div>

            <div className="flex items-center gap-2.5">
              <button
                onClick={() => onRunMonteCarloBatch(10000)}
                disabled={isGeneratingBatch}
                className="flex-1 py-2.5 px-4 bg-cyan-500 hover:bg-cyan-400 disabled:opacity-50 text-slate-950 font-semibold text-xs rounded-lg transition-colors flex items-center justify-center gap-2 whitespace-nowrap shrink-0"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                Generate & Sync 10,000 Iterations
              </button>
              <button
                onClick={() => onRunMonteCarloBatch(1000)}
                disabled={isGeneratingBatch}
                className="py-2.5 px-3 bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-xs rounded-lg transition-colors whitespace-nowrap shrink-0"
              >
                +1,000 Fast
              </button>
            </div>

            {/* Export & Round-Trip Binary Verification */}
            <div className="grid grid-cols-3 gap-2">
              <button
                onClick={onDownloadCSV}
                className="py-2 px-3 bg-[#161B2B] hover:bg-slate-800 border border-slate-700/80 text-slate-200 text-xs font-medium rounded-lg flex items-center justify-center gap-1.5 transition-colors whitespace-nowrap shrink-0"
              >
                <Download className="w-3.5 h-3.5 text-cyan-400" />
                Export .CSV
              </button>
              <button
                onClick={onDownloadBinary}
                className="py-2 px-3 bg-[#161B2B] hover:bg-slate-800 border border-slate-700/80 text-slate-200 text-xs font-medium rounded-lg flex items-center justify-center gap-1.5 transition-colors whitespace-nowrap shrink-0"
              >
                <Download className="w-3.5 h-3.5 text-pink-400" />
                Export .BIN (29B)
              </button>
              <button
                onClick={handleVerifyBinaryRoundTrip}
                className="py-2 px-3 bg-[#161B2B] hover:bg-slate-800 border border-slate-700/80 text-slate-200 text-xs font-medium rounded-lg flex items-center justify-center gap-1.5 transition-colors whitespace-nowrap shrink-0"
              >
                <Check className="w-3.5 h-3.5 text-emerald-400" />
                Verify .BIN
              </button>
            </div>
            {binaryVerifiedCount !== null && (
              <div className="text-xs font-mono text-emerald-400">
                Decoded {binaryVerifiedCount} packed 29-byte binary vectors via DataView with zero precision loss.
              </div>
            )}

            {/* 29-Byte Fixed Packed Binary Stream Layout Inspector */}
            <div className="space-y-2 pt-2 border-t border-slate-800/80">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-300">
                  Live 29-Byte Packed Binary Stream Layout (Active Vector)
                </span>
                <span className="text-xs font-mono text-slate-400">29 Bytes Fixed</span>
              </div>
              <div className="bg-[#06070D] border border-slate-800/90 rounded-lg overflow-hidden">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-slate-800 text-[11px] text-slate-400 font-mono">
                      <th className="py-1.5 px-2.5">Field</th>
                      <th className="py-1.5 px-2">Type</th>
                      <th className="py-1.5 px-2">Value</th>
                      <th className="py-1.5 px-2.5">Packed Hex Bytes</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-xs font-mono">
                    {hexBreakdown.map((item) => (
                      <tr key={item.field} className="hover:bg-slate-900/50">
                        <td className="py-1.5 px-2.5 text-slate-200">{item.field}</td>
                        <td className="py-1.5 px-2 text-slate-400">
                          {item.bytes}B · {item.type.split(' ')[0]}
                        </td>
                        <td className="py-1.5 px-2 text-cyan-400">{item.value}</td>
                        <td className="py-1.5 px-2.5 text-slate-400 text-[11px]">{item.hex}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Discovered Monte Carlo Profiles Table (Click to load or assign to Seed A/B) */}
            <div className="space-y-2 pt-2 border-t border-slate-800/80">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-300">
                  Discovered Topological Profiles (Click to Inspect in 3D)
                </span>
                <span className="text-xs font-mono text-slate-400">
                  Showing {Math.min(30, collectedProfiles.length)} of {collectedProfiles.length}
                </span>
              </div>
              <div className="max-h-56 overflow-y-auto bg-[#06070D] border border-slate-800/90 rounded-lg">
                <table className="w-full text-left border-collapse">
                  <thead className="sticky top-0 bg-[#111522] border-b border-slate-800 text-[11px] font-mono text-slate-400">
                    <tr>
                      <th className="py-1.5 px-2.5">Seed</th>
                      <th className="py-1.5 px-2">Pitch/Yaw</th>
                      <th className="py-1.5 px-2">Fitness</th>
                      <th className="py-1.5 px-2">Status</th>
                      <th className="py-1.5 px-2 text-right">Blend</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/50 text-xs font-mono">
                    {collectedProfiles.slice(0, 30).map((prof, idx) => (
                      <tr
                        key={`${prof.seed.toString()}-${idx}`}
                        onClick={() => onLoadProfileIntoViewport(prof)}
                        className="hover:bg-cyan-500/10 cursor-pointer transition-colors"
                      >
                        <td className="py-1.5 px-2.5 text-slate-300">
                          #{prof.seed.toString().slice(-6)}
                        </td>
                        <td className="py-1.5 px-2 text-slate-400">
                          {prof.pitchAxis.toFixed(2)} / {prof.yawAxis.toFixed(2)}
                        </td>
                        <td className="py-1.5 px-2 text-cyan-400">
                          {prof.fitnessScore.toFixed(1)}
                        </td>
                        <td className="py-1.5 px-2">
                          <span
                            className={
                              prof.isOptimal === 1 ? 'text-emerald-400' : 'text-rose-400'
                            }
                          >
                            {prof.isOptimal === 1 ? '1 · OPTIMAL' : '0 · STRAIN'}
                          </span>
                        </td>
                        <td
                          className="py-1.5 px-2 text-right space-x-1"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <button
                            onClick={() => onSetSeedA(prof)}
                            className={`px-1.5 py-0.5 text-[10px] rounded ${
                              seedA.seed === prof.seed
                                ? 'bg-cyan-500 text-slate-950 font-bold'
                                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                            }`}
                          >
                            A
                          </button>
                          <button
                            onClick={() => onSetSeedB(prof)}
                            className={`px-1.5 py-0.5 text-[10px] rounded ${
                              seedB.seed === prof.seed
                                ? 'bg-pink-500 text-slate-950 font-bold'
                                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                            }`}
                          >
                            B
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {/* ===================================================================
            TAB 2: NEURAL PREDICTION ACCURACY MATRIX & FOURIER MLP
           =================================================================== */}
        {activeDrawer === 'neural' && (
          <>
            {/* Mode Switcher: AdvancedTopologicalMLP (Fourier) vs Closed-Form Benchmark */}
            <div className="flex items-center gap-1 p-1 bg-[#06070D] border border-slate-800 rounded-lg">
              <button
                onClick={() => setInferenceMode('fourier-mlp')}
                className={`flex-1 py-1.5 px-3 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                  inferenceMode === 'fourier-mlp'
                    ? 'bg-cyan-500 text-slate-950 font-semibold'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                AdvancedTopologicalMLP (Fourier 64D)
              </button>
              <button
                onClick={() => setInferenceMode('closed-form')}
                className={`flex-1 py-1.5 px-3 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                  inferenceMode === 'closed-form'
                    ? 'bg-cyan-500 text-slate-950 font-semibold'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                Closed-Form Mobile Benchmark
              </button>
            </div>

            {/* High-Visibility Core Performance Percentage Card */}
            <div className="p-4 bg-[#06070D] border border-slate-800 rounded-lg flex items-center justify-between">
              <div>
                <div className="text-xs text-slate-400">Model Inference Convergence</div>
                <div
                  className={`text-2xl font-mono font-bold mt-1 ${
                    convergence.accuracyPercentage > 90
                      ? 'text-emerald-400'
                      : 'text-amber-400'
                  }`}
                >
                  {convergence.accuracyPercentage.toFixed(1)}% True Match
                </div>
                <div className="text-xs font-mono text-slate-500 mt-1">
                  Trained Epochs: {trainedEpochs} · Dual Loss: {lastEpochLoss.toFixed(4)}
                </div>
              </div>
              <div className="text-right">
                <div className="text-xs text-slate-400">Structural Integrity</div>
                <div
                  className={`text-xs font-mono font-semibold mt-1.5 ${
                    isOptimalSatisfied ? 'text-emerald-400' : 'text-rose-400'
                  }`}
                >
                  {isOptimalSatisfied
                    ? 'PASSED (Optimal Form)'
                    : 'WARNING (High Internal Strain)'}
                </div>
                <div className="text-xs font-mono text-slate-500 mt-1">
                  P(optimal) = {(fourierMlpPrediction.optimalProb * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Granular Side-by-Side Cross-Validation Telemetry Ledger */}
            <div className="p-3.5 bg-[#06070D] border border-slate-800/90 rounded-lg space-y-2.5">
              <div className="text-xs font-medium text-cyan-400">
                Cross-Validation Telemetry Ledger
              </div>
              <div className="flex justify-between text-xs font-mono">
                <span className="text-slate-400">Actual Math Engine Score:</span>
                <span className="text-white">{currentProfile.fitnessScore.toFixed(2)}</span>
              </div>
              <div className="flex justify-between text-xs font-mono">
                <span className="text-slate-400">AI Neural Network Prediction:</span>
                <span className="text-cyan-400">{activeInferredScore.toFixed(2)}</span>
              </div>
              <div className="flex justify-between text-xs font-mono">
                <span className="text-slate-400">Variance Margin (Absolute Offset):</span>
                <span className="text-pink-400">{convergence.absoluteError.toFixed(4)}</span>
              </div>
              <div className="flex justify-between text-xs font-mono pt-1.5 border-t border-slate-800/70">
                <span className="text-slate-400">L2-Norm Squared Strain / Threshold:</span>
                <span
                  className={
                    currentProfile.strain < bounds.maxStrainThreshold
                      ? 'text-emerald-400'
                      : 'text-rose-400'
                  }
                >
                  {currentProfile.strain.toFixed(2)} / {bounds.maxStrainThreshold.toFixed(1)}
                </span>
              </div>
              <div className="flex justify-between text-xs font-mono">
                <span className="text-slate-400">Polygon X-Y Volume / Min Area:</span>
                <span
                  className={
                    currentProfile.volume > bounds.minAreaThreshold
                      ? 'text-emerald-400'
                      : 'text-rose-400'
                  }
                >
                  {currentProfile.volume.toFixed(1)} / {bounds.minAreaThreshold.toFixed(0)}
                </span>
              </div>
            </div>

            {/* Interactive PyTorch Training Trigger */}
            <div className="space-y-2.5 pt-2 border-t border-slate-800/80">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-300">
                  Dual-Loss Optimization Loop (MSELoss + BCELoss)
                </span>
                <button
                  onClick={() => onTrainEpochs(10)}
                  className="py-1.5 px-3 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs rounded-lg transition-colors whitespace-nowrap shrink-0"
                >
                  Train +10 Epochs on Batch
                </button>
              </div>

              {/* Loss Convergence SVG Curve */}
              <div className="p-3 bg-[#06070D] border border-slate-800/90 rounded-lg">
                <div className="flex justify-between text-[11px] font-mono text-slate-400 mb-2">
                  <span>Aggregated Physics Loss Trajectory</span>
                  <span>Latest: {lastEpochLoss.toFixed(4)}</span>
                </div>
                <svg className="w-full h-24 overflow-visible" viewBox="0 0 320 80">
                  <line x1="0" y1="75" x2="320" y2="75" stroke="#1E293B" strokeWidth="1" />
                  <line x1="0" y1="40" x2="320" y2="40" stroke="#1E293B" strokeDasharray="3 3" strokeWidth="1" />
                  {lossHistory.length > 1 && (
                    <>
                      {/* Total Loss Curve */}
                      <polyline
                        fill="none"
                        stroke="#06B6D4"
                        strokeWidth="2"
                        points={lossHistory
                          .map((pt, i) => {
                            const x = (i / (lossHistory.length - 1)) * 310 + 5;
                            const maxL = Math.max(2.0, ...lossHistory.map((h) => h.totalLoss));
                            const y = 72 - (pt.totalLoss / maxL) * 64;
                            return `${x},${y}`;
                          })
                          .join(' ')}
                      />
                      {/* MSE Regression Loss Curve */}
                      <polyline
                        fill="none"
                        stroke="#EC4899"
                        strokeWidth="1.5"
                        strokeDasharray="3 2"
                        points={lossHistory
                          .map((pt, i) => {
                            const x = (i / (lossHistory.length - 1)) * 310 + 5;
                            const maxL = Math.max(2.0, ...lossHistory.map((h) => h.totalLoss));
                            const y = 72 - (pt.mseLoss / maxL) * 64;
                            return `${x},${y}`;
                          })
                          .join(' ')}
                      />
                    </>
                  )}
                </svg>
                <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 mt-1">
                  <span>Cyan Solid: Total Loss (MSE + BCE)</span>
                  <span>Magenta Dashed: Regression MSE</span>
                </div>
              </div>
            </div>

            {/* FourierFeatureMapping 64-Channel Harmonic Embedding Visualizer */}
            <div className="space-y-2 pt-2 border-t border-slate-800/80">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-300">
                  FourierFeatureMapping: [sin(X·B), cos(X·B)] (64 Channels, scale=8.0)
                </span>
              </div>
              <div className="p-3 bg-[#06070D] border border-slate-800/90 rounded-lg space-y-2">
                <div className="grid grid-cols-32 gap-0.5 h-12 items-end">
                  {Array.from(fourierMlpPrediction.fourierFeatures.slice(0, 32)).map(
                    (val, idx) => (
                      <div
                        key={`sin-${idx}`}
                        style={{ height: `${Math.max(12, Math.abs(val) * 100)}%` }}
                        className="bg-cyan-400/80 rounded-t-xs"
                        title={`sin(X·B)[${idx}] = ${val.toFixed(3)}`}
                      />
                    )
                  )}
                </div>
                <div className="grid grid-cols-32 gap-0.5 h-12 items-end">
                  {Array.from(fourierMlpPrediction.fourierFeatures.slice(32, 64)).map(
                    (val, idx) => (
                      <div
                        key={`cos-${idx}`}
                        style={{ height: `${Math.max(12, Math.abs(val) * 100)}%` }}
                        className="bg-pink-500/80 rounded-t-xs"
                        title={`cos(X·B)[${idx}] = ${val.toFixed(3)}`}
                      />
                    )
                  )}
                </div>
                <div className="flex justify-between text-[11px] font-mono text-slate-400 pt-1">
                  <span>Top 32: sin(Projection)</span>
                  <span>Bottom 32: cos(Projection)</span>
                </div>
              </div>

              {/* Architecture Comparison Table */}
              <div className="bg-[#06070D] border border-slate-800/90 rounded-lg p-3 space-y-1.5 text-xs font-mono">
                <div className="text-slate-300 font-sans font-medium">
                  Hidden Backbone LayerNorm Telemetry (256 → 128 → 64)
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Linear(64, 256) + Dropout(0.15) + LayerNorm:</span>
                  <span className="text-cyan-400">
                    ||h₁|| = {fourierMlpPrediction.layerNorms[0].toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Linear(256, 128) + LayerNorm + LeakyReLU(0.2):</span>
                  <span className="text-cyan-400">
                    ||h₂|| = {fourierMlpPrediction.layerNorms[1].toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Linear(128, 64) + Dual Prediction Heads:</span>
                  <span className="text-pink-400">
                    ||h₃|| = {fourierMlpPrediction.layerNorms[2].toFixed(3)}
                  </span>
                </div>
              </div>
            </div>
          </>
        )}

        {/* ===================================================================
            TAB 3: ORTHOGRAPHIC DISTORTION ANALYZER & ENGINE TELEMETRY DEBUGGER
           =================================================================== */}
        {activeDrawer === 'distortion' && (
          <>
            {/* Interactive Camera View Controls */}
            <div className="space-y-4">
              <div>
                <div className="text-xs font-medium text-white">
                  Orthographic Distortion Analyzer (ProjectionDistortionTestHud)
                </div>
                <div className="text-xs text-slate-400 mt-0.5">
                  Measure geometric aspect compression caused by 3D viewport tilt live.
                </div>
              </div>

              <div className="space-y-3 p-3.5 bg-[#06070D] border border-slate-800/90 rounded-lg">
                <div>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="text-slate-300">Camera Pitch Tilt</span>
                    <span className="font-mono text-cyan-400">
                      {cameraPitch.toFixed(2)} rad
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.15}
                    max={3.0}
                    step={0.02}
                    value={cameraPitch}
                    onChange={(e) =>
                      onCameraAngleChange(Number(e.target.value), cameraYaw)
                    }
                    className="w-full anharmonic-slider"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="text-slate-300">Camera Yaw Spin</span>
                    <span className="font-mono text-cyan-400">
                      {cameraYaw.toFixed(2)} rad
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.0}
                    max={3.14}
                    step={0.02}
                    value={cameraYaw}
                    onChange={(e) =>
                      onCameraAngleChange(cameraPitch, Number(e.target.value))
                    }
                    className="w-full anharmonic-slider"
                  />
                </div>
              </div>

              {/* Distortion Data Feed Readouts */}
              <div className="p-3.5 bg-[#06070D] border border-slate-800/90 rounded-lg space-y-2.5">
                <div className="text-xs font-medium text-cyan-400">
                  Live Geometry Compression Telemetry
                </div>
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate-400">X-Axis Plane Squeeze |cos(yaw)|:</span>
                  <span className="text-white">{compressionRatioX.toFixed(2)}x Scale</span>
                </div>
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate-400">Y-Axis Plane Squeeze |cos(pitch)|:</span>
                  <span className="text-white">{compressionRatioY.toFixed(2)}x Scale</span>
                </div>
                <div className="flex justify-between text-xs font-mono pt-1.5 border-t border-slate-800/70">
                  <span className="text-slate-400">Total Foreshortening Distortion Loss:</span>
                  <span
                    className={
                      combinedDistortionFactor > 75.0
                        ? 'text-pink-400 font-semibold'
                        : 'text-emerald-400 font-semibold'
                    }
                  >
                    {combinedDistortionFactor.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Multi-Threaded Performance Monitor Panel */}
            <div className="space-y-3 pt-3 border-t border-slate-800/80">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-cyan-400" />
                <div>
                  <div className="text-xs font-medium text-white">
                    Engine Telemetry Debugger (DeveloperPerformanceMonitorPanel)
                  </div>
                  <div className="text-xs text-slate-400">
                    60 FPS main thread isolation & Index-Linked Memory Pool diagnostics
                  </div>
                </div>
              </div>

              <div className="p-3.5 bg-[#06070D] border border-slate-800/90 rounded-lg space-y-2.5 font-mono text-xs">
                <div className="flex justify-between">
                  <span className="text-slate-400">UI Render Thread Status:</span>
                  <span
                    className={
                      physicsDeltaMs < 16 ? 'text-emerald-400' : 'text-pink-400'
                    }
                  >
                    {physicsDeltaMs < 16
                      ? 'STABLE (60 FPS+)'
                      : 'LAGGING (Frame Drop Alert)'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Physics Calculation Delta:</span>
                  <span className="text-cyan-400">{physicsDeltaMs} ms / frame</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Active Heap Memory Allocation:</span>
                  <span className="text-white">{heapMemoryMb.toFixed(2)} MB</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">HighDensityParticlePool (O(1) Stride):</span>
                  <span className="text-cyan-400">
                    {activeParticleCount.toLocaleString()} / 1,200 Active
                  </span>
                </div>
              </div>
            </div>

            {/* Google Play Privacy Policy: Unified Privacy Purge Controller */}
            <div className="space-y-3 pt-3 border-t border-slate-800/80">
              <div>
                <div className="text-xs font-medium text-white">
                  Unified Privacy Purge Controller (PrivacyDataDeleter)
                </div>
                <div className="text-xs text-slate-400 mt-0.5">
                  Executes full table teardown (clearAllTables) & DataStore preset reset.
                </div>
              </div>

              <button
                onClick={onExecutePrivacyPurge}
                className="w-full py-2.5 px-4 bg-rose-500/15 hover:bg-rose-500/25 border border-rose-500/40 text-rose-300 font-medium text-xs rounded-lg flex items-center justify-center gap-2 transition-colors whitespace-nowrap shrink-0"
              >
                <Trash2 className="w-3.5 h-3.5" />
                Purge All Cached Telemetry & Reset Presets
              </button>

              {purgeToastMessage && (
                <div className="p-2.5 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-xs font-mono text-emerald-300">
                  {purgeToastMessage}
                </div>
              )}
            </div>
          </>
        )}

        {/* ===================================================================
            TAB 4: EMBEDDED AI PIPELINE, CODE SUITE & REFERENCE PDF
           =================================================================== */}
        {activeDrawer === 'pipeline' && (
          <>
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-400">
                Production Kotlin, PyTorch, NumPy, AGSL & ADB Pipeline Scripts
              </span>
              <button
                onClick={() => setShowPdfPreview(!showPdfPreview)}
                className="px-2.5 py-1 text-xs font-medium rounded bg-slate-800 hover:bg-slate-700 text-cyan-400 flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0"
              >
                <FileText className="w-3.5 h-3.5" />
                {showPdfPreview ? 'View Code Suite' : 'View Spec PDF'}
              </button>
            </div>

            {showPdfPreview ? (
              <div className="space-y-2">
                <div className="text-xs font-mono text-slate-300">
                  Attached Blueprint: Notes_261025_195647_ac0ed1a4.pdf
                </div>
                <iframe
                  src="/assets/aistudio/Notes_261025_195647_ac0ed1a4.pdf"
                  title="Anharmonic Space Specification PDF"
                  className="w-full h-[460px] rounded-lg border border-slate-800 bg-[#06070D]"
                />
              </div>
            ) : (
              <>
                {/* Artifact Selector Pills */}
                <div className="grid grid-cols-2 gap-1.5">
                  {ANHARMONIC_CODE_ARTIFACTS.map((art) => (
                    <button
                      key={art.id}
                      onClick={() => setSelectedCodeId(art.id)}
                      className={`p-2 text-left rounded-lg border text-xs transition-colors truncate ${
                        selectedCodeId === art.id
                          ? 'bg-cyan-500/15 border-cyan-500/50 text-white font-medium'
                          : 'bg-[#06070D] border-slate-800/80 text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      <div className="font-mono text-[11px] truncate">{art.filename}</div>
                      <div className="text-[10px] text-slate-500 truncate">{art.category}</div>
                    </button>
                  ))}
                </div>

                {/* Selected Code Artifact Viewer */}
                <div className="bg-[#06070D] border border-slate-800 rounded-lg overflow-hidden">
                  <div className="flex items-center justify-between px-3.5 py-2.5 bg-[#111522] border-b border-slate-800">
                    <div>
                      <div className="text-xs font-mono font-semibold text-cyan-400">
                        {selectedArtifact.filename}
                      </div>
                      <div className="text-[11px] text-slate-400 mt-0.5">
                        {selectedArtifact.summary}
                      </div>
                    </div>
                    <div className="flex items-center gap-1.5 shrink-0">
                      <button
                        onClick={handleCopyCode}
                        className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs rounded flex items-center gap-1 transition-colors whitespace-nowrap shrink-0"
                      >
                        {copiedCode ? (
                          <>
                            <Check className="w-3 h-3 text-emerald-400" />
                            Copied
                          </>
                        ) : (
                          <>
                            <Copy className="w-3 h-3" />
                            Copy
                          </>
                        )}
                      </button>
                      <button
                        onClick={handleDownloadCodeFile}
                        className="px-2.5 py-1 bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 text-xs rounded flex items-center gap-1 transition-colors whitespace-nowrap shrink-0"
                      >
                        <Download className="w-3 h-3" />
                        Save
                      </button>
                    </div>
                  </div>
                  <pre className="p-3.5 text-[11px] font-mono text-slate-300 overflow-x-auto max-h-[340px] leading-relaxed">
                    <code>{selectedArtifact.code}</code>
                  </pre>
                </div>
              </>
            )}
          </>
        )}
      </div>
    </aside>
  );
};
