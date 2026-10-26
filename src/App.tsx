/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useMemo, useCallback, useEffect } from 'react';
import {
  BinaryDatasetPacker,
  DiscoveredTopologyProfile,
  FourierTopologicalMLP,
  GenerationBounds,
  MonteCarloTopologicalExplorer,
  evaluateLatentVectorPhysics,
  calculateConvergenceAccuracy,
} from './engine/topologicalCore';
import { SpatialViewport3D, ViewportMode } from './components/SpatialViewport3D';
import { LaboratoryDrawers, ActiveDrawerTab } from './components/LaboratoryDrawers';
import {
  ChevronDown,
  ChevronUp,
  Database,
  Sparkles,
  Sliders,
  FileCode,
  Play,
  Pause,
  RotateCcw,
  Grid,
  Shuffle,
} from 'lucide-react';

const DEFAULT_BOUNDS: GenerationBounds = {
  nodeCount: 4,
  targetStiffness: 0.5,
  maxStrainThreshold: 15.0,
  minAreaThreshold: 200.0,
};

export default function App() {
  // 1. Primary 3D Viewport Mode
  const [viewportMode, setViewportMode] = useState<ViewportMode>('null-axis-sdf');

  // 2. Latent Axis Vector Track Sliders (AiLatentSpaceExplorerPanel state)
  const [latentPitch, setLatentPitch] = useState<number>(1.18);
  const [latentYaw, setLatentYaw] = useState<number>(0.84);
  const [latentFreq, setLatentFreq] = useState<number>(2.1);
  const [latentScale, setLatentScale] = useState<number>(165.0);
  const [bounds, setBounds] = useState<GenerationBounds>(DEFAULT_BOUNDS);

  // 3. Viewport & Shader Controls
  const [cameraPitch, setCameraPitch] = useState<number>(1.22);
  const [cameraYaw, setCameraYaw] = useState<number>(0.45);
  const [terrainComplexity, setTerrainComplexity] = useState<number>(2.4);
  const [showWireframe, setShowWireframe] = useState<boolean>(true);
  const [animateWave, setAnimateWave] = useState<boolean>(true);
  const [isDockCollapsed, setIsDockCollapsed] = useState<boolean>(false);
  const [activeDrawer, setActiveDrawer] = useState<ActiveDrawerTab>('none');

  // 4. Persistent Engines (Monte Carlo Explorer & Fourier MLP)
  const explorer = useMemo(() => new MonteCarloTopologicalExplorer(), []);
  const fourierMlp = useMemo(() => new FourierTopologicalMLP(8.0, 421337n), []);

  // Initialize with a 1,000-sample Monte Carlo dataset so tables and Seed A/B are populated immediately
  const initialBatch = useMemo(
    () => explorer.generateBatch(1000, DEFAULT_BOUNDS, false),
    [explorer]
  );

  const [collectedProfiles, setCollectedProfiles] = useState<DiscoveredTopologyProfile[]>(
    initialBatch.profiles
  );
  const [totalGeneratedCount, setTotalGeneratedCount] = useState<number>(1000);
  const [optimalTotalCount, setOptimalTotalCount] = useState<number>(
    initialBatch.optimalCount
  );
  const [lastBatchTimeMs, setLastBatchTimeMs] = useState<number>(
    initialBatch.executionTimeMs
  );
  const [isGeneratingBatch, setIsGeneratingBatch] = useState<boolean>(false);
  const [keepOptimalOnly, setKeepOptimalOnly] = useState<boolean>(false);

  // Seed A and Seed B for Latent Space Vector Blending Interpolator
  const [seedA, setSeedA] = useState<DiscoveredTopologyProfile>(() => {
    const firstOptimal = initialBatch.profiles.find((p) => p.isOptimal === 1);
    return firstOptimal || initialBatch.profiles[0];
  });
  const [seedB, setSeedB] = useState<DiscoveredTopologyProfile>(() => {
    const lastOptimal = [...initialBatch.profiles]
      .reverse()
      .find((p) => p.isOptimal === 1 && p.seed !== seedA?.seed);
    return lastOptimal || initialBatch.profiles[Math.min(1, initialBatch.profiles.length - 1)];
  });
  const [blendAlpha, setBlendAlpha] = useState<number>(0.3);

  // Neural training epoch tracker
  const [trainedEpochs, setTrainedEpochs] = useState<number>(fourierMlp.trainedEpochs);
  const [lastEpochLoss, setLastEpochLoss] = useState<number>(fourierMlp.lastEpochLoss);
  const [lossHistory, setLossHistory] = useState(fourierMlp.lossHistory);

  // Live 60FPS Telemetry
  const [physicsDeltaMs, setPhysicsDeltaMs] = useState<number>(2);
  const [activeParticleCount, setActiveParticleCount] = useState<number>(1200);
  const [purgeToastMessage, setPurgeToastMessage] = useState<string | null>(null);

  // Keyboard shortcut 'H' to toggle Zen/Cinema mode dock collapse
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        (e.key === 'h' || e.key === 'H') &&
        document.activeElement?.tagName !== 'INPUT'
      ) {
        setIsDockCollapsed((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Effective latent parameters (uses interpolated Seed A <-> Seed B when in latent-interpolator mode)
  const effectivePitch =
    viewportMode === 'latent-interpolator'
      ? seedA.pitchAxis + (seedB.pitchAxis - seedA.pitchAxis) * blendAlpha
      : latentPitch;
  const effectiveYaw =
    viewportMode === 'latent-interpolator'
      ? seedA.yawAxis + (seedB.yawAxis - seedA.yawAxis) * blendAlpha
      : latentYaw;
  const effectiveFreq =
    viewportMode === 'latent-interpolator'
      ? seedA.timeFreq + (seedB.timeFreq - seedA.timeFreq) * blendAlpha
      : latentFreq;
  const effectiveScale =
    viewportMode === 'latent-interpolator'
      ? seedA.globalScale + (seedB.globalScale - seedA.globalScale) * blendAlpha
      : latentScale;

  // Evaluate ground-truth physics & Fourier MLP forward pass for current vector
  const currentProfile = useMemo(
    () =>
      evaluateLatentVectorPhysics(
        effectivePitch,
        effectiveYaw,
        effectiveFreq,
        effectiveScale,
        bounds
      ),
    [effectivePitch, effectiveYaw, effectiveFreq, effectiveScale, bounds]
  );

  const fourierMlpPrediction = useMemo(
    () =>
      fourierMlp.forward(
        effectivePitch,
        effectiveYaw,
        effectiveFreq,
        effectiveScale,
        bounds
      ),
    [
      fourierMlp,
      effectivePitch,
      effectiveYaw,
      effectiveFreq,
      effectiveScale,
      bounds,
      trainedEpochs,
    ]
  );

  const convergence = useMemo(
    () =>
      calculateConvergenceAccuracy(
        currentProfile.fitnessScore,
        fourierMlpPrediction.fitnessPred
      ),
    [currentProfile.fitnessScore, fourierMlpPrediction.fitnessPred]
  );

  // Handlers
  const handleRunMonteCarloBatch = useCallback(
    (iterations: number) => {
      setIsGeneratingBatch(true);
      setTimeout(() => {
        const res = explorer.generateBatch(iterations, bounds, keepOptimalOnly);
        setCollectedProfiles((prev) => [...res.profiles, ...prev].slice(0, 600));
        setTotalGeneratedCount((prev) => prev + iterations);
        setOptimalTotalCount((prev) => prev + res.optimalCount);
        setLastBatchTimeMs(res.executionTimeMs);
        setIsGeneratingBatch(false);
      }, 20);
    },
    [explorer, bounds, keepOptimalOnly]
  );

  const handleTrainEpochs = useCallback(
    (epochs: number) => {
      fourierMlp.trainStepOnDataset(collectedProfiles, epochs);
      setTrainedEpochs(fourierMlp.trainedEpochs);
      setLastEpochLoss(fourierMlp.lastEpochLoss);
      setLossHistory([...fourierMlp.lossHistory]);
    },
    [fourierMlp, collectedProfiles]
  );

  const handleLoadProfileIntoViewport = useCallback(
    (prof: DiscoveredTopologyProfile) => {
      setLatentPitch(Number(prof.pitchAxis.toFixed(2)));
      setLatentYaw(Number(prof.yawAxis.toFixed(2)));
      setLatentFreq(Number(prof.timeFreq.toFixed(2)));
      setLatentScale(Number(prof.globalScale.toFixed(1)));
      if (viewportMode === 'latent-interpolator') {
        setViewportMode('null-axis-sdf');
      }
    },
    [viewportMode]
  );

  const handleRandomizeOptimalVector = useCallback(() => {
    const optimalPool = collectedProfiles.filter((p) => p.isOptimal === 1);
    if (optimalPool.length > 0) {
      const pick = optimalPool[Math.floor(Math.random() * optimalPool.length)];
      handleLoadProfileIntoViewport(pick);
    } else {
      setLatentPitch(Number((Math.random() * 6.28).toFixed(2)));
      setLatentYaw(Number((Math.random() * 6.28).toFixed(2)));
      setLatentFreq(Number((0.5 + Math.random() * 4.0).toFixed(1)));
      setLatentScale(Number((90 + Math.random() * 180).toFixed(0)));
    }
  }, [collectedProfiles, handleLoadProfileIntoViewport]);

  const handleResetDefaults = useCallback(() => {
    setLatentPitch(1.18);
    setLatentYaw(0.84);
    setLatentFreq(2.1);
    setLatentScale(165.0);
    setBounds(DEFAULT_BOUNDS);
    setCameraPitch(1.22);
    setCameraYaw(0.45);
  }, []);

  const handleDownloadCSV = useCallback(() => {
    const csv = BinaryDatasetPacker.exportCSVString(collectedProfiles);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ai_topology_dataset.csv';
    a.click();
    URL.revokeObjectURL(url);
  }, [collectedProfiles]);

  const handleDownloadBinary = useCallback(() => {
    const buffer = BinaryDatasetPacker.packProfilesToBinary(collectedProfiles);
    const blob = new Blob([buffer], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'anharmonic_ai_training_set.bin';
    a.click();
    URL.revokeObjectURL(url);
  }, [collectedProfiles]);

  const handleExecutePrivacyPurge = useCallback(() => {
    setCollectedProfiles([]);
    setTotalGeneratedCount(0);
    setOptimalTotalCount(0);
    handleResetDefaults();
    setPurgeToastMessage(
      'Room SQLite telemetry tables & DataStore preferences purged (0 bytes cached).'
    );
    setTimeout(() => setPurgeToastMessage(null), 4000);
  }, [handleResetDefaults]);

  const handleCameraAngleChange = useCallback((pitch: number, yaw: number) => {
    setCameraPitch(pitch);
    setCameraYaw(yaw);
  }, []);

  const handleFrameTelemetry = useCallback((deltaMs: number, activeParticles: number) => {
    setPhysicsDeltaMs(deltaMs);
    setActiveParticleCount(activeParticles);
  }, []);

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-[#06070D] select-none">
      {/* =====================================================================
          1. STRICT 3-ZONE TOP BAR CONTRACT
          Zone 1: Brand Wordmark | Zone 2: 4 Single-Line Nav Links | Zone 3: 1 Primary Action
         ===================================================================== */}
      <header className="fixed top-0 left-0 right-0 z-30 h-12 flex items-center justify-between gap-8 px-6 bg-[#06070D]/95 border-b border-slate-800/80">
        {/* Zone 1: Single text element wordmark */}
        <a
          href="#top"
          onClick={(e) => {
            e.preventDefault();
            setViewportMode('null-axis-sdf');
          }}
          className="text-base font-bold tracking-tight text-white font-display whitespace-nowrap shrink-0"
        >
          Anharmonic Space
        </a>

        {/* Zone 2: 4 concise single-line text navigation links */}
        <nav className="hidden md:flex items-center gap-6 text-xs font-medium text-slate-400">
          <a
            href="#null-axis"
            onClick={(e) => {
              e.preventDefault();
              setViewportMode('null-axis-sdf');
            }}
            className={`transition-colors whitespace-nowrap shrink-0 py-1 ${
              viewportMode === 'null-axis-sdf'
                ? 'text-cyan-400 underline underline-offset-8 decoration-cyan-400 decoration-2'
                : 'hover:text-white'
            }`}
          >
            Null-Axis Lattice
          </a>
          <a
            href="#terrain"
            onClick={(e) => {
              e.preventDefault();
              setViewportMode('procedural-terrain');
            }}
            className={`transition-colors whitespace-nowrap shrink-0 py-1 ${
              viewportMode === 'procedural-terrain'
                ? 'text-cyan-400 underline underline-offset-8 decoration-cyan-400 decoration-2'
                : 'hover:text-white'
            }`}
          >
            Procedural Terrain
          </a>
          <a
            href="#interpolator"
            onClick={(e) => {
              e.preventDefault();
              setViewportMode('latent-interpolator');
            }}
            className={`transition-colors whitespace-nowrap shrink-0 py-1 ${
              viewportMode === 'latent-interpolator'
                ? 'text-cyan-400 underline underline-offset-8 decoration-cyan-400 decoration-2'
                : 'hover:text-white'
            }`}
          >
            Vector Blending
          </a>
          <a
            href="#fourier"
            onClick={(e) => {
              e.preventDefault();
              setViewportMode('fourier-manifold');
            }}
            className={`transition-colors whitespace-nowrap shrink-0 py-1 ${
              viewportMode === 'fourier-manifold'
                ? 'text-cyan-400 underline underline-offset-8 decoration-cyan-400 decoration-2'
                : 'hover:text-white'
            }`}
          >
            Fourier Manifold
          </a>
        </nav>

        {/* Zone 3: 1 primary action */}
        <div className="flex items-center gap-3 shrink-0">
          <button
            onClick={handleDownloadBinary}
            className="px-3.5 py-1.5 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-lg transition-colors whitespace-nowrap shrink-0"
          >
            Export .BIN Dataset
          </button>
        </div>
      </header>

      {/* =====================================================================
          2. FULL-BLEED 3D SPATIAL VIEWPORT (100% Unobstructed Center)
         ===================================================================== */}
      <main className="w-full h-full pt-12">
        <SpatialViewport3D
          mode={viewportMode}
          latentPitch={latentPitch}
          latentYaw={latentYaw}
          latentFreq={latentFreq}
          latentScale={latentScale}
          bounds={bounds}
          cameraPitch={cameraPitch}
          cameraYaw={cameraYaw}
          onCameraAngleChange={handleCameraAngleChange}
          blendAlpha={blendAlpha}
          seedA={seedA}
          seedB={seedB}
          fourierFeatures={fourierMlpPrediction.fourierFeatures}
          terrainComplexity={terrainComplexity}
          showWireframe={showWireframe}
          animateWave={animateWave}
          onFrameTelemetry={handleFrameTelemetry}
        />
      </main>

      {/* =====================================================================
          3. DISCREET TOP-LEFT SCIENTIFIC TELEMETRY OVERLAY (Unboxed Metadata)
         ===================================================================== */}
      <div className="fixed top-16 left-6 z-20 pointer-events-none max-w-md space-y-1">
        <div className="text-xs font-mono text-slate-400 flex flex-wrap items-center gap-1.5">
          <span className="text-white font-semibold">Topological Matrix Core v1.0.0</span>
          <span aria-hidden="true">·</span>
          <span>Tokens: {totalGeneratedCount.toLocaleString()}</span>
          <span aria-hidden="true">·</span>
          <span>{physicsDeltaMs} ms/frame</span>
        </div>

        <div className="text-xs font-mono flex flex-wrap items-center gap-2">
          <span className="text-slate-300">
            Fitness:{' '}
            <strong className="text-cyan-400">
              {currentProfile.fitnessScore.toFixed(2)}
            </strong>
          </span>
          <span className="text-slate-600" aria-hidden="true">
            ·
          </span>
          <span className="text-slate-300">
            MLP Pred:{' '}
            <strong className="text-pink-400">
              {fourierMlpPrediction.fitnessPred.toFixed(2)}
            </strong>
          </span>
          <span className="text-slate-600" aria-hidden="true">
            ·
          </span>
          <span className="text-slate-300">
            Match:{' '}
            <strong className="text-emerald-400">
              {convergence.accuracyPercentage.toFixed(1)}%
            </strong>
          </span>
        </div>

        <div className="text-xs font-mono flex items-center gap-2 pt-0.5">
          <span
            className={
              currentProfile.isOptimal === 1
                ? 'text-emerald-400 font-medium'
                : 'text-rose-400 font-medium'
            }
          >
            {currentProfile.isOptimal === 1
              ? '● PASSED (Optimal Form)'
              : '▲ WARNING (High Internal Strain)'}
          </span>
          <span className="text-slate-600" aria-hidden="true">
            ·
          </span>
          <span className="text-slate-400">
            Strain: {currentProfile.strain.toFixed(2)} / Vol:{' '}
            {currentProfile.volume.toFixed(0)}
          </span>
        </div>
      </div>

      {/* =====================================================================
          4. SLIDE-OVER LABORATORY TELEMETRY & CODE DRAWERS
         ===================================================================== */}
      <LaboratoryDrawers
        activeDrawer={activeDrawer}
        onCloseDrawer={() => setActiveDrawer('none')}
        currentProfile={currentProfile}
        bounds={bounds}
        onUpdateBounds={setBounds}
        onLoadProfileIntoViewport={handleLoadProfileIntoViewport}
        onSetSeedA={setSeedA}
        onSetSeedB={setSeedB}
        seedA={seedA}
        seedB={seedB}
        collectedProfiles={collectedProfiles}
        totalGeneratedCount={totalGeneratedCount}
        optimalTotalCount={optimalTotalCount}
        isGeneratingBatch={isGeneratingBatch}
        lastBatchTimeMs={lastBatchTimeMs}
        keepOptimalOnly={keepOptimalOnly}
        onToggleKeepOptimalOnly={() => setKeepOptimalOnly((p) => !p)}
        onRunMonteCarloBatch={handleRunMonteCarloBatch}
        onDownloadCSV={handleDownloadCSV}
        onDownloadBinary={handleDownloadBinary}
        fourierMlpPrediction={fourierMlpPrediction}
        trainedEpochs={trainedEpochs}
        lastEpochLoss={lastEpochLoss}
        lossHistory={lossHistory}
        onTrainEpochs={handleTrainEpochs}
        cameraPitch={cameraPitch}
        cameraYaw={cameraYaw}
        onCameraAngleChange={handleCameraAngleChange}
        physicsDeltaMs={physicsDeltaMs}
        activeParticleCount={activeParticleCount}
        onExecutePrivacyPurge={handleExecutePrivacyPurge}
        purgeToastMessage={purgeToastMessage}
      />

      {/* =====================================================================
          5. UNIFIED COLLAPSIBLE BOTTOM CONTROL DOCK (Zero Viewport Occlusion)
         ===================================================================== */}
      <div
        className="fixed left-1/2 -translate-x-1/2 bottom-4 z-30 w-[94vw] max-w-[940px] transition-transform duration-200"
        style={{
          transform: isDockCollapsed
            ? 'translateY(calc(100% - 28px))'
            : 'translateY(0)',
        }}
      >
        {/* Collapse / Expand Handle attached to top center edge of dock */}
        <div className="flex justify-center">
          <button
            onClick={() => setIsDockCollapsed((p) => !p)}
            className="px-4 py-1 bg-[#111522] hover:bg-[#161B2B] text-slate-300 hover:text-white text-xs font-mono border border-b-0 border-slate-800 rounded-t-lg flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0"
          >
            {isDockCollapsed ? (
              <>
                <ChevronUp className="w-3.5 h-3.5 text-cyan-400" />
                Show Latent & Telemetry Controls (H)
              </>
            ) : (
              <>
                <ChevronDown className="w-3.5 h-3.5 text-cyan-400" />
                Hide Controls (H)
              </>
            )}
          </button>
        </div>

        {/* Main Bottom Dock Surface (Solid rgba surface, no backdrop-filter over WebGL) */}
        <div className="bg-[#0B0E17]/95 border border-slate-800 rounded-xl p-4 shadow-2xl space-y-3.5">
          {/* Top Row of Dock: Laboratory Drawer Tabs + Quick Scene Actions */}
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-800/80">
            {/* Drawer Toggle Buttons */}
            <div className="flex flex-wrap items-center gap-1.5">
              <button
                onClick={() =>
                  setActiveDrawer(activeDrawer === 'tokenizer' ? 'none' : 'tokenizer')
                }
                className={`px-3 py-1.5 text-xs font-medium rounded-lg flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0 ${
                  activeDrawer === 'tokenizer'
                    ? 'bg-cyan-500 text-slate-950 font-semibold'
                    : 'bg-[#161B2B] text-slate-300 hover:text-white border border-slate-800'
                }`}
              >
                <Database className="w-3.5 h-3.5" />
                Monte Carlo & Binary ({totalGeneratedCount.toLocaleString()})
              </button>

              <button
                onClick={() =>
                  setActiveDrawer(activeDrawer === 'neural' ? 'none' : 'neural')
                }
                className={`px-3 py-1.5 text-xs font-medium rounded-lg flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0 ${
                  activeDrawer === 'neural'
                    ? 'bg-cyan-500 text-slate-950 font-semibold'
                    : 'bg-[#161B2B] text-slate-300 hover:text-white border border-slate-800'
                }`}
              >
                <Sparkles className="w-3.5 h-3.5" />
                Neural Matrix ({convergence.accuracyPercentage.toFixed(1)}%)
              </button>

              <button
                onClick={() =>
                  setActiveDrawer(activeDrawer === 'distortion' ? 'none' : 'distortion')
                }
                className={`px-3 py-1.5 text-xs font-medium rounded-lg flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0 ${
                  activeDrawer === 'distortion'
                    ? 'bg-cyan-500 text-slate-950 font-semibold'
                    : 'bg-[#161B2B] text-slate-300 hover:text-white border border-slate-800'
                }`}
              >
                <Sliders className="w-3.5 h-3.5" />
                Distortion & Telemetry
              </button>

              <button
                onClick={() =>
                  setActiveDrawer(activeDrawer === 'pipeline' ? 'none' : 'pipeline')
                }
                className={`px-3 py-1.5 text-xs font-medium rounded-lg flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0 ${
                  activeDrawer === 'pipeline'
                    ? 'bg-cyan-500 text-slate-950 font-semibold'
                    : 'bg-[#161B2B] text-slate-300 hover:text-white border border-slate-800'
                }`}
              >
                <FileCode className="w-3.5 h-3.5" />
                Kotlin / PyTorch Code Suite
              </button>
            </div>

            {/* Scene Utility Controls */}
            <div className="flex items-center gap-1.5">
              <button
                onClick={() => setAnimateWave((w) => !w)}
                className="px-2.5 py-1.5 bg-[#161B2B] hover:bg-slate-800 border border-slate-800 text-slate-200 text-xs font-medium rounded-lg flex items-center gap-1 transition-colors whitespace-nowrap shrink-0"
                title="Toggle continuous time wave (t)"
              >
                {animateWave ? (
                  <>
                    <Pause className="w-3.5 h-3.5 text-cyan-400" />
                    Wave Active
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 text-slate-400" />
                    Wave Paused
                  </>
                )}
              </button>

              {viewportMode === 'procedural-terrain' && (
                <button
                  onClick={() => setShowWireframe((w) => !w)}
                  className={`px-2.5 py-1.5 border text-xs font-medium rounded-lg flex items-center gap-1 transition-colors whitespace-nowrap shrink-0 ${
                    showWireframe
                      ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-300'
                      : 'bg-[#161B2B] border-slate-800 text-slate-400'
                  }`}
                >
                  <Grid className="w-3.5 h-3.5" />
                  Mesh Grid
                </button>
              )}

              <button
                onClick={handleRandomizeOptimalVector}
                className="px-2.5 py-1.5 bg-[#161B2B] hover:bg-slate-800 border border-slate-800 text-slate-200 text-xs font-medium rounded-lg flex items-center gap-1 transition-colors whitespace-nowrap shrink-0"
                title="Sample a discovered optimal Monte Carlo seed"
              >
                <Shuffle className="w-3.5 h-3.5 text-pink-400" />
                Sample Optimal
              </button>

              <button
                onClick={handleResetDefaults}
                className="p-1.5 bg-[#161B2B] hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-white rounded-lg transition-colors"
                title="Reset latent axes to default"
                aria-label="Reset latent axes"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Bottom Row of Dock: 4 Real-Time Latent Axis Sliders OR Vector Blending Interpolator */}
          {viewportMode === 'latent-interpolator' ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
              <div className="p-2.5 bg-[#06070D] border border-slate-800/90 rounded-lg font-mono text-xs">
                <div className="text-cyan-400 font-semibold">
                  Seed A (#{seedA.seed.toString().slice(-6)})
                </div>
                <div className="text-slate-400 text-[11px] mt-0.5">
                  P:{seedA.pitchAxis.toFixed(2)} · Y:{seedA.yawAxis.toFixed(2)} · F:
                  {seedA.timeFreq.toFixed(1)} · Fit:{seedA.fitnessScore.toFixed(1)}
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-slate-300 font-medium">
                    Latent Space Linear Interpolation (α)
                  </span>
                  <span className="font-mono text-cyan-400">
                    {(blendAlpha * 100).toFixed(0)}% Seed B
                  </span>
                </div>
                <input
                  type="range"
                  min={0.0}
                  max={1.0}
                  step={0.01}
                  value={blendAlpha}
                  onChange={(e) => setBlendAlpha(Number(e.target.value))}
                  className="w-full anharmonic-slider"
                />
              </div>

              <div className="p-2.5 bg-[#06070D] border border-slate-800/90 rounded-lg font-mono text-xs text-right">
                <div className="text-pink-400 font-semibold">
                  Seed B (#{seedB.seed.toString().slice(-6)})
                </div>
                <div className="text-slate-400 text-[11px] mt-0.5">
                  P:{seedB.pitchAxis.toFixed(2)} · Y:{seedB.yawAxis.toFixed(2)} · F:
                  {seedB.timeFreq.toFixed(1)} · Fit:{seedB.fitnessScore.toFixed(1)}
                </div>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {/* Slider 1: Latent Pitch Axis */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300">Latent Pitch Axis</span>
                  <span className="font-mono text-cyan-400">
                    {latentPitch.toFixed(2)} rad
                  </span>
                </div>
                <input
                  type="range"
                  min={0.0}
                  max={6.28}
                  step={0.02}
                  value={latentPitch}
                  onChange={(e) => setLatentPitch(Number(e.target.value))}
                  className="w-full anharmonic-slider"
                />
              </div>

              {/* Slider 2: Latent Yaw Axis */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300">Latent Yaw Axis</span>
                  <span className="font-mono text-cyan-400">
                    {latentYaw.toFixed(2)} rad
                  </span>
                </div>
                <input
                  type="range"
                  min={0.0}
                  max={6.28}
                  step={0.02}
                  value={latentYaw}
                  onChange={(e) => setLatentYaw(Number(e.target.value))}
                  className="w-full anharmonic-slider"
                />
              </div>

              {/* Slider 3: Latent Step Frequency OR Terrain Complexity */}
              {viewportMode === 'procedural-terrain' ? (
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300">Harmonic Complexity</span>
                    <span className="font-mono text-pink-400">
                      {terrainComplexity.toFixed(1)}x
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.5}
                    max={6.0}
                    step={0.1}
                    value={terrainComplexity}
                    onChange={(e) => setTerrainComplexity(Number(e.target.value))}
                    className="w-full anharmonic-slider anharmonic-slider-magenta"
                  />
                </div>
              ) : (
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300">Latent Step Frequency</span>
                    <span className="font-mono text-cyan-400">
                      {latentFreq.toFixed(1)} Hz
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.1}
                    max={10.0}
                    step={0.1}
                    value={latentFreq}
                    onChange={(e) => setLatentFreq(Number(e.target.value))}
                    className="w-full anharmonic-slider"
                  />
                </div>
              )}

              {/* Slider 4: Latent Global Scale */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300">Latent Global Scale</span>
                  <span className="font-mono text-cyan-400">
                    {latentScale.toFixed(0)} u
                  </span>
                </div>
                <input
                  type="range"
                  min={50.0}
                  max={300.0}
                  step={1.0}
                  value={latentScale}
                  onChange={(e) => setLatentScale(Number(e.target.value))}
                  className="w-full anharmonic-slider"
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
