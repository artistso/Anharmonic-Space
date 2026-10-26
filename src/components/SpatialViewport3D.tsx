import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import {
  DiscoveredTopologyProfile,
  GenerationBounds,
  HighDensityParticlePool,
  synthesizeNodeCoordinates,
  evaluateLatentVectorPhysics,
} from '../engine/topologicalCore';

export type ViewportMode =
  | 'null-axis-sdf'
  | 'procedural-terrain'
  | 'latent-interpolator'
  | 'fourier-manifold';

interface SpatialViewport3DProps {
  mode: ViewportMode;
  latentPitch: number;
  latentYaw: number;
  latentFreq: number;
  latentScale: number;
  bounds: GenerationBounds;
  cameraPitch: number;
  cameraYaw: number;
  onCameraAngleChange: (pitch: number, yaw: number) => void;
  blendAlpha: number;
  seedA: DiscoveredTopologyProfile;
  seedB: DiscoveredTopologyProfile;
  fourierFeatures: Float32Array;
  terrainComplexity: number;
  showWireframe: boolean;
  animateWave: boolean;
  onFrameTelemetry: (deltaMs: number, activeParticles: number) => void;
}

// Pre-allocated module-scope math helpers (Zero GC in render loop)
const _vec3 = new THREE.Vector3();
const _vec3B = new THREE.Vector3();
const _mat4 = new THREE.Matrix4();
const _quat = new THREE.Quaternion();
const _euler = new THREE.Euler();
const _scale = new THREE.Vector3(1, 1, 1);
const _color = new THREE.Color();
const _colorCyan = new THREE.Color(0x06b6d4);
const _colorMagenta = new THREE.Color(0xec4899);
const _colorEmerald = new THREE.Color(0x10b981);

export const SpatialViewport3D: React.FC<SpatialViewport3DProps> = ({
  mode,
  latentPitch,
  latentYaw,
  latentFreq,
  latentScale,
  bounds,
  cameraPitch,
  cameraYaw,
  onCameraAngleChange,
  blendAlpha,
  seedA,
  seedB,
  fourierFeatures,
  terrainComplexity,
  showWireframe,
  animateWave,
  onFrameTelemetry,
}) => {
  const canvasMountRef = useRef<HTMLDivElement | null>(null);

  // Store mutable props in ref so animation loop reads latest state without tearing down WebGL
  const stateRef = useRef({
    mode,
    latentPitch,
    latentYaw,
    latentFreq,
    latentScale,
    bounds,
    cameraPitch,
    cameraYaw,
    blendAlpha,
    seedA,
    seedB,
    fourierFeatures,
    terrainComplexity,
    showWireframe,
    animateWave,
    onCameraAngleChange,
    onFrameTelemetry,
  });

  useEffect(() => {
    stateRef.current = {
      mode,
      latentPitch,
      latentYaw,
      latentFreq,
      latentScale,
      bounds,
      cameraPitch,
      cameraYaw,
      blendAlpha,
      seedA,
      seedB,
      fourierFeatures,
      terrainComplexity,
      showWireframe,
      animateWave,
      onCameraAngleChange,
      onFrameTelemetry,
    };
  });

  useEffect(() => {
    const container = canvasMountRef.current;
    if (!container) return;

    // 1. Software WebGL detection from GPU renderer string
    const isSoftwareGl = (() => {
      const probeCanvas = document.createElement('canvas');
      const gl = probeCanvas.getContext('webgl2');
      if (!gl) return true;
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
      const gpu = String(gl.getParameter(param));
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
    })();

    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;

    const renderer = new THREE.WebGLRenderer({
      antialias: !isSoftwareGl,
      powerPreference: 'high-performance',
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.12;
    renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = !isSoftwareGl;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.setClearColor(0x06070d, 1);

    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x06070d);
    scene.fog = new THREE.FogExp2(0x06070d, 0.0014);

    // 2. Studio Environment Reflection (RoomEnvironment)
    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    scene.environment = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;
    pmremGenerator.dispose();

    // 3. Camera & OrbitControls
    const camera = new THREE.PerspectiveCamera(50, width / height, 1, 2500);
    camera.position.set(0, 140, 390);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = !isSoftwareGl;
    controls.dampingFactor = 0.06;
    controls.maxDistance = 950;
    controls.minDistance = 80;
    controls.target.set(0, 0, 0);

    let userInteractingWithOrbit = false;
    let lastSyncedPitch = stateRef.current.cameraPitch;
    let lastSyncedYaw = stateRef.current.cameraYaw;

    controls.addEventListener('start', () => {
      userInteractingWithOrbit = true;
    });
    controls.addEventListener('end', () => {
      userInteractingWithOrbit = false;
      // Sync spherical angles back to Orthographic Distortion HUD
      const offset = _vec3.copy(camera.position).sub(controls.target);
      const radius = Math.max(1, offset.length());
      const pitch = Math.acos(THREE.MathUtils.clamp(offset.y / radius, -1, 1));
      let yaw = Math.atan2(offset.x, offset.z);
      if (yaw < 0) yaw += Math.PI;
      lastSyncedPitch = pitch;
      lastSyncedYaw = yaw;
      stateRef.current.onCameraAngleChange(
        Number(pitch.toFixed(2)),
        Number(yaw.toFixed(2))
      );
    });

    // 4. Balanced 3-Point Studio Lighting Rig
    const hemiLight = new THREE.HemisphereLight(0x38bdf8, 0x0f172a, 0.75);
    scene.add(hemiLight);

    const keyLight = new THREE.DirectionalLight(0xfffbeb, 2.2);
    keyLight.position.set(180, 280, 200);
    if (!isSoftwareGl) {
      keyLight.castShadow = true;
      keyLight.shadow.mapSize.set(1024, 1024);
      keyLight.shadow.camera.near = 20;
      keyLight.shadow.camera.far = 900;
      const d = 260;
      keyLight.shadow.camera.left = -d;
      keyLight.shadow.camera.right = d;
      keyLight.shadow.camera.top = d;
      keyLight.shadow.camera.bottom = -d;
      keyLight.shadow.bias = -0.0005;
      keyLight.shadow.normalBias = 0.02;
    }
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0xec4899, 1.45);
    rimLight.position.set(-220, 120, -180);
    scene.add(rimLight);

    // Ground Reference Grid + Contact Shadow Plane
    const gridHelper = new THREE.GridHelper(900, 36, 0x06b6d4, 0x161b2b);
    gridHelper.position.y = -150;
    (gridHelper.material as THREE.Material).transparent = true;
    (gridHelper.material as THREE.Material).opacity = 0.35;
    scene.add(gridHelper);

    const shadowPlaneGeo = new THREE.PlaneGeometry(500, 500);
    const shadowPlaneMat = new THREE.MeshBasicMaterial({
      color: 0x020307,
      transparent: true,
      opacity: 0.42,
      depthWrite: false,
    });
    const shadowPlane = new THREE.Mesh(shadowPlaneGeo, shadowPlaneMat);
    shadowPlane.rotation.x = -Math.PI / 2;
    shadowPlane.position.y = -149.8;
    scene.add(shadowPlane);

    // =========================================================================
    // MODE 1: FLOATING NULL-AXIS & SDF FLUID LATTICE GROUP
    // =========================================================================
    const nullAxisGroup = new THREE.Group();
    scene.add(nullAxisGroup);

    const MAX_NODES = 8;
    const nodeGroups: THREE.Group[] = [];
    const nodeCoreMats: THREE.MeshPhysicalMaterial[] = [];

    // Composite 6-part procedural model for each Floating Null-Axis Gimbal Node
    const outerRingGeo = new THREE.TorusGeometry(13, 1.2, 16, 48);
    const innerRingGeo = new THREE.TorusGeometry(9, 0.9, 12, 36);
    const coreSphereGeo = new THREE.OctahedronGeometry(4.8, 2);
    const axisSpindleGeo = new THREE.CylinderGeometry(0.45, 0.45, 34, 12);
    const collarGeo = new THREE.CylinderGeometry(2.4, 2.8, 3.2, 16);

    const metalDarkMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      metalness: 0.9,
      roughness: 0.2,
    });
    const brushedGoldMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      metalness: 0.85,
      roughness: 0.25,
    });
    const spindleMat = new THREE.MeshStandardMaterial({
      color: 0x94a3b8,
      metalness: 0.75,
      roughness: 0.45,
    });

    for (let i = 0; i < MAX_NODES; i++) {
      const g = new THREE.Group();
      // 1. Primary silhouette: Outer bevelled gimbal ring
      const outerRing = new THREE.Mesh(outerRingGeo, metalDarkMat);
      g.add(outerRing);

      // 2. Secondary read: Orthogonal inner harmonic ring
      const innerRing = new THREE.Mesh(innerRingGeo, brushedGoldMat);
      innerRing.rotation.y = Math.PI / 2;
      g.add(innerRing);

      // 3. Accent detail: Vertical & Horizontal null-axis spindles
      const spindleY = new THREE.Mesh(axisSpindleGeo, spindleMat);
      g.add(spindleY);
      const spindleX = new THREE.Mesh(axisSpindleGeo, spindleMat);
      spindleX.rotation.z = Math.PI / 2;
      g.add(spindleX);

      // 4. Top/Bottom collars
      const collarTop = new THREE.Mesh(collarGeo, metalDarkMat);
      collarTop.position.y = 13;
      g.add(collarTop);
      const collarBot = new THREE.Mesh(collarGeo, metalDarkMat);
      collarBot.position.y = -13;
      g.add(collarBot);

      // 5. Phase-reactive glowing core crystal
      const coreMat = new THREE.MeshPhysicalMaterial({
        color: 0x06b6d4,
        emissive: 0x06b6d4,
        emissiveIntensity: 0.65,
        roughness: 0.15,
        metalness: 0.1,
        clearcoat: 1.0,
      });
      const coreMesh = new THREE.Mesh(coreSphereGeo, coreMat);
      g.add(coreMesh);

      nodeCoreMats.push(coreMat);
      nodeGroups.push(g);
      nullAxisGroup.add(g);
    }

    // Structural Chain Tube / Line Segments & Polygon Cross-Product Volume Fill
    const chainLinePositions = new Float32Array((MAX_NODES + 1) * 3);
    const chainLineGeo = new THREE.BufferGeometry();
    chainLineGeo.setAttribute(
      'position',
      new THREE.BufferAttribute(chainLinePositions, 3)
    );
    const chainLineMat = new THREE.LineBasicMaterial({
      color: 0x22d3ee,
      transparent: true,
      opacity: 0.9,
    });
    const chainLine = new THREE.Line(chainLineGeo, chainLineMat);
    nullAxisGroup.add(chainLine);

    // Interior Polygon Volume Mesh
    const polyPositions = new Float32Array(MAX_NODES * 3);
    const polyGeo = new THREE.BufferGeometry();
    polyGeo.setAttribute('position', new THREE.BufferAttribute(polyPositions, 3));
    const polyMat = new THREE.MeshBasicMaterial({
      color: 0x06b6d4,
      transparent: true,
      opacity: 0.14,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const polyMesh = new THREE.Mesh(polyGeo, polyMat);
    nullAxisGroup.add(polyMesh);

    // 40-Ring Tracer History Buffer (InstancedMesh)
    const TRACER_HISTORY = 40;
    const TOTAL_TRACERS = MAX_NODES * TRACER_HISTORY;
    const tracerGeo = new THREE.TorusGeometry(4.2, 0.35, 8, 20);
    const tracerMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.45,
    });
    const tracerInstanced = new THREE.InstancedMesh(
      tracerGeo,
      tracerMat,
      TOTAL_TRACERS
    );
    tracerInstanced.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    tracerInstanced.frustumCulled = false;
    nullAxisGroup.add(tracerInstanced);

    const tracerHistoryBuffer = new Float32Array(TOTAL_TRACERS * 3);
    let tracerWriteHead = 0;

    // HighDensityParticlePool (O(1) Index-Linked Memory Pool for SDF Fluid)
    const PARTICLE_COUNT = 1200;
    const particlePool = new HighDensityParticlePool(PARTICLE_COUNT);
    particlePool.resetPool(PARTICLE_COUNT, 105);

    const sdfParticleGeo = new THREE.OctahedronGeometry(1.4, 0);
    const sdfParticleMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.25,
      metalness: 0.7,
    });
    const sdfInstanced = new THREE.InstancedMesh(
      sdfParticleGeo,
      sdfParticleMat,
      PARTICLE_COUNT
    );
    sdfInstanced.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    sdfInstanced.frustumCulled = false;
    nullAxisGroup.add(sdfInstanced);

    // =========================================================================
    // MODE 2: 3D PROCEDURAL TERRAIN MATRIX (50x50 Mesh + AGSL-Equivalent Shader)
    // =========================================================================
    const terrainGroup = new THREE.Group();
    terrainGroup.visible = false;
    scene.add(terrainGroup);

    const GRID_RES = 50;
    const terrainGeo = new THREE.PlaneGeometry(520, 520, GRID_RES - 1, GRID_RES - 1);
    terrainGeo.rotateX(-Math.PI / 2);

    const terrainUniforms = {
      uGlobalTime: { value: 0 },
      uMaxHeight: { value: 90.0 },
    };

    // Custom ShaderMaterial faithfully implementing TerrainColorizer.agsl
    const terrainShaderMat = new THREE.ShaderMaterial({
      uniforms: terrainUniforms,
      wireframe: false,
      side: THREE.DoubleSide,
      vertexShader: `
        varying vec3 vWorldPos;
        varying vec3 vNormalDir;
        varying vec2 vUvCoord;
        void main() {
          vUvCoord = uv;
          vNormalDir = normalize(normalMatrix * normal);
          vec4 worldPosition = modelMatrix * vec4(position, 1.0);
          vWorldPos = worldPosition.xyz;
          gl_Position = projectionMatrix * viewMatrix * worldPosition;
        }
      `,
      fragmentShader: `
        uniform float uGlobalTime;
        uniform float uMaxHeight;
        varying vec3 vWorldPos;
        varying vec3 vNormalDir;
        varying vec2 vUvCoord;

        void main() {
          float normElevation = clamp((vWorldPos.y + uMaxHeight) / (2.0 * uMaxHeight), 0.0, 1.0);
          float slopeStrain = abs(sin(vWorldPos.y * 0.065 + uGlobalTime * 0.4));

          // 1. Deep cosmic valley palette
          vec3 colorPalette = vec3(0.02, 0.03, 0.08);

          // 2. Safe flat terrain zones -> vibrant luminous Cyan via absolute Sine waves
          float valleyMask = smoothstep(0.0, 0.55, 1.0 - normElevation);
          colorPalette += vec3(0.0, 0.45 + abs(sin(uGlobalTime)) * 0.35, 1.0) * valleyMask * 0.65;

          // 3. Steep ridges & high-strain cliffs -> sharp neon Magenta via Cosine waves
          float ridgeMask = smoothstep(0.32, 0.88, normElevation * (0.45 + slopeStrain));
          colorPalette += vec3(1.0, 0.08, 0.45 + abs(cos(uGlobalTime * 1.2)) * 0.45) * ridgeMask * 0.95;

          // Subtle topographic contour lines
          float contour = abs(fract(vWorldPos.y * 0.08) - 0.5);
          float gridLine = smoothstep(0.04, 0.0, contour);
          colorPalette += vec3(0.15, 0.85, 0.95) * gridLine * 0.28;

          // Rim lighting
          float lightDot = max(0.25, dot(vNormalDir, normalize(vec3(0.5, 1.0, 0.4))));
          gl_FragColor = vec4(colorPalette * (0.65 + 0.45 * lightDot), 1.0);
        }
      `,
    });

    const terrainMesh = new THREE.Mesh(terrainGeo, terrainShaderMat);
    terrainMesh.position.y = -20;
    terrainGroup.add(terrainMesh);

    const terrainWireMat = new THREE.MeshBasicMaterial({
      color: 0x22d3ee,
      wireframe: true,
      transparent: true,
      opacity: 0.18,
    });
    const terrainWireMesh = new THREE.Mesh(terrainGeo, terrainWireMat);
    terrainWireMesh.position.y = -19.6;
    terrainGroup.add(terrainWireMesh);

    // =========================================================================
    // MODE 3: LATENT SPACE VECTOR BLENDING INTERPOLATOR GROUP
    // =========================================================================
    const interpolatorGroup = new THREE.Group();
    interpolatorGroup.visible = false;
    scene.add(interpolatorGroup);

    const GHOST_STEPS = 11;
    const ghostLines: THREE.LineLoop[] = [];
    const ghostMats: THREE.LineBasicMaterial[] = [];

    for (let s = 0; s < GHOST_STEPS; s++) {
      const gGeo = new THREE.BufferGeometry();
      gGeo.setAttribute(
        'position',
        new THREE.BufferAttribute(new Float32Array(MAX_NODES * 3), 3)
      );
      const isCenter = s === Math.floor(GHOST_STEPS / 2);
      const gMat = new THREE.LineBasicMaterial({
        color: isCenter ? 0x10b981 : 0x06b6d4,
        transparent: true,
        opacity: isCenter ? 0.85 : 0.35,
      });
      const loop = new THREE.LineLoop(gGeo, gMat);
      ghostLines.push(loop);
      ghostMats.push(gMat);
      interpolatorGroup.add(loop);
    }

    // Active Interpolated Form Highlight Group
    const activeBlendGeo = new THREE.BufferGeometry();
    activeBlendGeo.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(MAX_NODES * 3), 3)
    );
    const activeBlendMat = new THREE.LineBasicMaterial({
      color: 0xffffff,
      linewidth: 2,
    });
    const activeBlendLoop = new THREE.LineLoop(activeBlendGeo, activeBlendMat);
    interpolatorGroup.add(activeBlendLoop);

    const activeBlendNodes: THREE.Mesh[] = [];
    const blendNodeGeo = new THREE.SphereGeometry(5.5, 16, 16);
    const blendNodeMat = new THREE.MeshStandardMaterial({
      color: 0x06b6d4,
      emissive: 0x06b6d4,
      emissiveIntensity: 0.7,
      roughness: 0.2,
      metalness: 0.5,
    });
    for (let i = 0; i < MAX_NODES; i++) {
      const m = new THREE.Mesh(blendNodeGeo, blendNodeMat);
      activeBlendNodes.push(m);
      interpolatorGroup.add(m);
    }

    // =========================================================================
    // MODE 4: FOURIER FEATURE MAPPING (4 -> 64 HARMONIC MANIFOLD) GROUP
    // =========================================================================
    const fourierGroup = new THREE.Group();
    fourierGroup.visible = false;
    scene.add(fourierGroup);

    const FOURIER_BARS = 64;
    const fourierBarGeo = new THREE.BoxGeometry(3.2, 1, 3.2);
    fourierBarGeo.translate(0, 0.5, 0); // Pivot at bottom-center
    const fourierBarMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.2,
      metalness: 0.8,
    });
    const fourierInstanced = new THREE.InstancedMesh(
      fourierBarGeo,
      fourierBarMat,
      FOURIER_BARS
    );
    fourierInstanced.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    fourierInstanced.frustumCulled = false;
    fourierGroup.add(fourierInstanced);

    // Pre-warm shaders on GPU
    if (!isSoftwareGl) {
      renderer.compile(scene, camera);
    }

    // =========================================================================
    // ANIMATION & RENDER LOOP (with Software WebGL Fence Sync Pacing)
    // =========================================================================
    const clock = new THREE.Clock();
    const currentNodeCoords = new Float32Array(MAX_NODES * 3);
    const tempBlendCoords = new Float32Array(MAX_NODES * 3);
    let frameId = 0;
    let frameCount = 0;

    function afterDraw(cb: () => void) {
      const gl = renderer.getContext() as WebGL2RenderingContext;
      const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();
      const poll = () => {
        const state = sync && gl.getSyncParameter(sync, gl.SYNC_STATUS);
        if (state === gl.UNSIGNALED) {
          frameId = window.setTimeout(poll, 5);
        } else {
          if (sync) gl.deleteSync(sync);
          frameId = window.setTimeout(cb, 80);
        }
      };
      frameId = window.setTimeout(poll, 5);
    }

    function tick() {
      const tStart = performance.now();
      const delta = Math.min(clock.getDelta(), 0.1);
      const elapsed = clock.getElapsedTime();
      const st = stateRef.current;

      // Smoothly sync camera position if user adjusted HUD camera sliders
      if (
        !userInteractingWithOrbit &&
        (Math.abs(st.cameraPitch - lastSyncedPitch) > 0.02 ||
          Math.abs(st.cameraYaw - lastSyncedYaw) > 0.02)
      ) {
        lastSyncedPitch = THREE.MathUtils.damp(
          lastSyncedPitch,
          st.cameraPitch,
          6,
          delta
        );
        lastSyncedYaw = THREE.MathUtils.damp(
          lastSyncedYaw,
          st.cameraYaw,
          6,
          delta
        );
        const dist = camera.position.distanceTo(controls.target);
        const sinP = Math.sin(Math.max(0.15, Math.min(Math.PI - 0.15, lastSyncedPitch)));
        const cosP = Math.cos(Math.max(0.15, Math.min(Math.PI - 0.15, lastSyncedPitch)));
        camera.position.set(
          controls.target.x + dist * sinP * Math.sin(lastSyncedYaw),
          controls.target.y + dist * cosP,
          controls.target.z + dist * sinP * Math.cos(lastSyncedYaw)
        );
        camera.lookAt(controls.target);
      }

      controls.update();

      const phaseOffset = st.animateWave ? elapsed * 0.85 : 0;
      const nodeCount = Math.min(MAX_NODES, Math.max(3, st.bounds.nodeCount));

      // Compute current 3D node coordinates
      synthesizeNodeCoordinates(
        st.latentPitch,
        st.latentYaw,
        st.latentFreq,
        st.latentScale,
        nodeCount,
        phaseOffset,
        currentNodeCoords
      );

      // Toggle active 3D mode groups
      nullAxisGroup.visible =
        st.mode === 'null-axis-sdf' || st.mode === 'fourier-manifold';
      terrainGroup.visible = st.mode === 'procedural-terrain';
      interpolatorGroup.visible = st.mode === 'latent-interpolator';
      fourierGroup.visible = st.mode === 'fourier-manifold';

      if (nullAxisGroup.visible) {
        // Update floating null-axis gimbal nodes
        for (let i = 0; i < MAX_NODES; i++) {
          const g = nodeGroups[i];
          if (i < nodeCount) {
            g.visible = true;
            const nx = currentNodeCoords[i * 3];
            const ny = currentNodeCoords[i * 3 + 1];
            const nz = currentNodeCoords[i * 3 + 2];
            g.position.set(nx, ny, nz);
            g.rotation.x = st.latentPitch + elapsed * 0.5 + i * 0.4;
            g.rotation.y = st.latentYaw + elapsed * 0.4;

            // Trigonometric color harmonics (Sine -> Cyan, Cosine -> Magenta)
            const harmonicRatio = 0.5 + 0.5 * Math.cos(i * st.latentFreq + elapsed);
            _color.lerpColors(_colorCyan, _colorMagenta, harmonicRatio);
            nodeCoreMats[i].color.copy(_color);
            nodeCoreMats[i].emissive.copy(_color);
          } else {
            g.visible = false;
          }
        }

        // Update closed chain line
        const linePosAttr = chainLineGeo.attributes.position as THREE.BufferAttribute;
        for (let i = 0; i <= nodeCount; i++) {
          const srcIdx = (i % nodeCount) * 3;
          linePosAttr.setXYZ(
            i,
            currentNodeCoords[srcIdx],
            currentNodeCoords[srcIdx + 1],
            currentNodeCoords[srcIdx + 2]
          );
        }
        chainLineGeo.setDrawRange(0, nodeCount + 1);
        linePosAttr.needsUpdate = true;

        // Update polygon cross-product volume surface (fan triangulation from node 0)
        const polyPosAttr = polyGeo.attributes.position as THREE.BufferAttribute;
        for (let i = 0; i < nodeCount; i++) {
          polyPosAttr.setXYZ(
            i,
            currentNodeCoords[i * 3],
            currentNodeCoords[i * 3 + 1],
            currentNodeCoords[i * 3 + 2]
          );
        }
        const indices: number[] = [];
        for (let i = 1; i < nodeCount - 1; i++) {
          indices.push(0, i, i + 1);
        }
        polyGeo.setIndex(indices);
        polyPosAttr.needsUpdate = true;

        // Record tracer ring history
        for (let i = 0; i < nodeCount; i++) {
          const slot = (i * TRACER_HISTORY + tracerWriteHead) * 3;
          tracerHistoryBuffer[slot] = currentNodeCoords[i * 3];
          tracerHistoryBuffer[slot + 1] = currentNodeCoords[i * 3 + 1];
          tracerHistoryBuffer[slot + 2] = currentNodeCoords[i * 3 + 2];
        }
        tracerWriteHead = (tracerWriteHead + 1) % TRACER_HISTORY;

        let instanceIdx = 0;
        for (let i = 0; i < MAX_NODES; i++) {
          for (let h = 0; h < TRACER_HISTORY; h++) {
            if (i < nodeCount) {
              const slot = (i * TRACER_HISTORY + h) * 3;
              const age = (tracerWriteHead - h + TRACER_HISTORY) % TRACER_HISTORY;
              const s = Math.max(0.05, 1.0 - age / TRACER_HISTORY);
              _vec3.set(
                tracerHistoryBuffer[slot],
                tracerHistoryBuffer[slot + 1],
                tracerHistoryBuffer[slot + 2]
              );
              _euler.set(elapsed + h * 0.1, i * 0.5, 0);
              _quat.setFromEuler(_euler);
              _scale.setScalar(s);
              _mat4.compose(_vec3, _quat, _scale);
              tracerInstanced.setMatrixAt(instanceIdx, _mat4);
              _color.lerpColors(_colorCyan, _colorMagenta, age / TRACER_HISTORY);
              tracerInstanced.setColorAt(instanceIdx, _color);
            } else {
              _scale.setScalar(0);
              _mat4.compose(_vec3.set(0, 0, 0), _quat.identity(), _scale);
              tracerInstanced.setMatrixAt(instanceIdx, _mat4);
            }
            instanceIdx++;
          }
        }
        tracerInstanced.instanceMatrix.needsUpdate = true;
        if (tracerInstanced.instanceColor) {
          tracerInstanced.instanceColor.needsUpdate = true;
        }

        // Step HighDensityParticlePool Verlet SDF fluid simulation
        particlePool.stepVerletSdfPhysics(
          currentNodeCoords,
          nodeCount,
          elapsed,
          st.bounds.targetStiffness
        );

        for (let p = 0; p < particlePool.activeCount; p++) {
          const base = particlePool.activeIndices[p] * particlePool.stride;
          const cx = particlePool.dataPool[base];
          const cy = particlePool.dataPool[base + 1];
          const vx = cx - particlePool.dataPool[base + 2];
          const vy = cy - particlePool.dataPool[base + 3];
          const speed = Math.min(1.0, Math.sqrt(vx * vx + vy * vy) * 0.45);

          const cz =
            Math.sin(cx * 0.025 + st.latentYaw) *
            Math.cos(cy * 0.025 + st.latentPitch) *
            35.0;

          _vec3.set(cx, cy, cz);
          _euler.set(elapsed + p * 0.02, elapsed * 0.7, 0);
          _quat.setFromEuler(_euler);
          _scale.setScalar(0.75 + speed * 0.9);
          _mat4.compose(_vec3, _quat, _scale);
          sdfInstanced.setMatrixAt(p, _mat4);

          _color.lerpColors(_colorCyan, _colorMagenta, speed);
          sdfInstanced.setColorAt(p, _color);
        }
        sdfInstanced.instanceMatrix.needsUpdate = true;
        if (sdfInstanced.instanceColor) {
          sdfInstanced.instanceColor.needsUpdate = true;
        }
      }

      // =======================================================================
      // MODE 2 UPDATE: 3D PROCEDURAL TERRAIN MATRIX
      // =======================================================================
      if (terrainGroup.visible) {
        terrainUniforms.uGlobalTime.value = elapsed;
        terrainUniforms.uMaxHeight.value = Math.max(25, st.latentScale * 0.45);
        terrainWireMesh.visible = st.showWireframe;

        const posAttr = terrainGeo.attributes.position as THREE.BufferAttribute;
        const pitchFreq = (0.6 + st.latentPitch * 0.4) * 0.022;
        const yawFreq = (0.6 + st.latentYaw * 0.4) * 0.022;
        const amplitude = st.latentScale * 0.36;
        const complexity = st.terrainComplexity;

        for (let i = 0; i < posAttr.count; i++) {
          const wx = posAttr.getX(i);
          const wz = posAttr.getZ(i);
          const sampleX = wx * pitchFreq + phaseOffset * 0.6;
          const sampleZ = wz * yawFreq + phaseOffset * 0.4;

          let hy = Math.sin(sampleX) * Math.cos(sampleZ) * amplitude;
          hy +=
            Math.cos(sampleX * complexity) *
            Math.sin(sampleZ * complexity) *
            (amplitude * 0.25);
          posAttr.setY(i, hy);
        }
        posAttr.needsUpdate = true;
        terrainGeo.computeVertexNormals();
      }

      // =======================================================================
      // MODE 3 UPDATE: LATENT SPACE VECTOR BLENDING INTERPOLATOR
      // =======================================================================
      if (interpolatorGroup.visible) {
        const a = st.seedA;
        const b = st.seedB;

        for (let s = 0; s < GHOST_STEPS; s++) {
          const tAlpha = s / (GHOST_STEPS - 1);
          const pPitch = THREE.MathUtils.lerp(a.pitchAxis, b.pitchAxis, tAlpha);
          const pYaw = THREE.MathUtils.lerp(a.yawAxis, b.yawAxis, tAlpha);
          const pFreq = THREE.MathUtils.lerp(a.timeFreq, b.timeFreq, tAlpha);
          const pScale = THREE.MathUtils.lerp(a.globalScale, b.globalScale, tAlpha) * 0.55;

          const evalStep = evaluateLatentVectorPhysics(
            pPitch,
            pYaw,
            pFreq,
            pScale,
            st.bounds
          );

          const xOffset = (tAlpha - 0.5) * 380;
          const posAttr = ghostLines[s].geometry.attributes
            .position as THREE.BufferAttribute;

          for (let n = 0; n < nodeCount; n++) {
            posAttr.setXYZ(
              n,
              evalStep.coordinates[n * 3] + xOffset,
              evalStep.coordinates[n * 3 + 1],
              evalStep.coordinates[n * 3 + 2]
            );
          }
          ghostLines[s].geometry.setDrawRange(0, nodeCount);
          posAttr.needsUpdate = true;
          ghostMats[s].color.copy(
            evalStep.isOptimal === 1 ? _colorCyan : _colorMagenta
          );
        }

        // Highlighted user-scrubbed blendAlpha configuration
        const curPitch = THREE.MathUtils.lerp(a.pitchAxis, b.pitchAxis, st.blendAlpha);
        const curYaw = THREE.MathUtils.lerp(a.yawAxis, b.yawAxis, st.blendAlpha);
        const curFreq = THREE.MathUtils.lerp(a.timeFreq, b.timeFreq, st.blendAlpha);
        const curScale =
          THREE.MathUtils.lerp(a.globalScale, b.globalScale, st.blendAlpha) * 0.55;

        synthesizeNodeCoordinates(
          curPitch,
          curYaw,
          curFreq,
          curScale,
          nodeCount,
          phaseOffset,
          tempBlendCoords
        );
        const activeXOffset = (st.blendAlpha - 0.5) * 380;
        const actAttr = activeBlendGeo.attributes.position as THREE.BufferAttribute;

        for (let i = 0; i < MAX_NODES; i++) {
          if (i < nodeCount) {
            const bx = tempBlendCoords[i * 3] + activeXOffset;
            const by = tempBlendCoords[i * 3 + 1];
            const bz = tempBlendCoords[i * 3 + 2];
            actAttr.setXYZ(i, bx, by, bz);
            activeBlendNodes[i].visible = true;
            activeBlendNodes[i].position.set(bx, by, bz);
          } else {
            activeBlendNodes[i].visible = false;
          }
        }
        activeBlendGeo.setDrawRange(0, nodeCount);
        actAttr.needsUpdate = true;
      }

      // =======================================================================
      // MODE 4 UPDATE: FOURIER 64-DIMENSIONAL HARMONIC MANIFOLD RING
      // =======================================================================
      if (fourierGroup.visible) {
        const ringRadius = 195;
        for (let f = 0; f < FOURIER_BARS; f++) {
          const angle = (f / FOURIER_BARS) * Math.PI * 2 + elapsed * 0.12;
          const val = st.fourierFeatures[f] || 0;
          const barHeight = 8 + Math.abs(val) * 75;

          _vec3.set(
            Math.cos(angle) * ringRadius,
            -110,
            Math.sin(angle) * ringRadius
          );
          _euler.set(0, -angle, 0);
          _quat.setFromEuler(_euler);
          _scale.set(1, barHeight, 1);
          _mat4.compose(_vec3, _quat, _scale);
          fourierInstanced.setMatrixAt(f, _mat4);

          // First 32 = sin(Projection) in Cyan, Second 32 = cos(Projection) in Magenta
          if (f < 32) {
            _color.lerpColors(_colorCyan, _colorEmerald, Math.abs(val));
          } else {
            _color.lerpColors(_colorMagenta, _colorCyan, 1.0 - Math.abs(val));
          }
          fourierInstanced.setColorAt(f, _color);
        }
        fourierInstanced.instanceMatrix.needsUpdate = true;
        if (fourierInstanced.instanceColor) {
          fourierInstanced.instanceColor.needsUpdate = true;
        }
      }

      renderer.render(scene, camera);

      const execDeltaMs = Math.max(1, Math.round(performance.now() - tStart));
      frameCount++;
      if (frameCount % 12 === 0) {
        st.onFrameTelemetry(execDeltaMs, particlePool.activeCount);
      }

      if (isSoftwareGl) {
        afterDraw(tick);
      } else {
        frameId = requestAnimationFrame(tick);
      }
    }

    frameId = requestAnimationFrame(tick);

    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth || window.innerWidth;
      const h = container.clientHeight || window.innerHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(frameId);
      clearTimeout(frameId);
      controls.dispose();
      renderer.dispose();
    };
  }, []);

  return (
    <div
      ref={canvasMountRef}
      className="w-full h-full cursor-crosshair select-none"
      aria-label="Anharmonic Space 3D Spatial Viewport"
    />
  );
};
