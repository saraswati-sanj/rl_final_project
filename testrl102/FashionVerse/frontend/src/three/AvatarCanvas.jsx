import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

const COLOR_MAP = {
  black: 0x1a1a1a,
  white: 0xf0f0f0,
  navy: 0x16263b,
  grey: 0x6e7480,
  beige: 0xd9c5a0,
  olive: 0x556b2f,
  maroon: 0x781226,
  royal_blue: 0x205090,
  forest_green: 0x1f6634,
  burnt_orange: 0xc44d18,
  dusty_rose: 0xd69992,
  lavender: 0xb388eb,
  teal: 0x008080,
  mustard: 0xcca01a,
  cream: 0xfdf8e2,
  charcoal: 0x2f3640,
  rust: 0xb03a2e,
  sage_green: 0x8fbc8f,
  wine: 0x5b1e31,
  sky_blue: 0x70b8e8,
  coral: 0xf76c5e,
  tan: 0xd2b48c,
  blush: 0xde5d83,
  slate: 0x607d8b,
  indigo: 0x3f51b5,
};

export default function AvatarCanvas({ outfit = [], isVRActive = false, onToggleVR }) {
  const mountRef = useRef(null);
  const meshesRef = useRef({});
  const isDraggingRef = useRef(false);
  const prevMouseRef = useRef({ x: 0, y: 0 });
  const [vrSupported, setVrSupported] = useState(false);
  const [vrStatusMsg, setVrStatusMsg] = useState('');

  // Check WebXR VR support
  useEffect(() => {
    if (navigator.xr) {
      navigator.xr.isSessionSupported('immersive-vr')
        .then((supported) => setVrSupported(supported))
        .catch(() => setVrSupported(false));
    }
  }, []);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 400;
    const height = container.clientHeight || 500;

    // 1. Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0e111a);
    scene.fog = new THREE.FogExp2(0x0e111a, 0.08);

    // 2. Camera setup
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(0, 1.2, 3.4);

    // 3. Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    // 4. Lighting (High-end Studio setup)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xfff5ea, 1.4);
    keyLight.position.set(2, 4, 3);
    keyLight.castShadow = true;
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x8899ff, 0.6);
    fillLight.position.set(-3, 2, 2);
    scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0x6c63ff, 1.2);
    rimLight.position.set(0, 3, -3);
    scene.add(rimLight);

    // 5. Studio Platform (Circular pedestal)
    const platformGeo = new THREE.CylinderGeometry(1.2, 1.3, 0.1, 48);
    const platformMat = new THREE.MeshStandardMaterial({
      color: 0x151928,
      roughness: 0.4,
      metalness: 0.3,
    });
    const platform = new THREE.Mesh(platformGeo, platformMat);
    platform.position.y = -0.85;
    platform.receiveShadow = true;
    scene.add(platform);

    const ringGeo = new THREE.RingGeometry(1.22, 1.27, 48);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x6c63ff, side: THREE.DoubleSide });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = -0.79;
    scene.add(ring);

    // 6. Build Avatar Hierarchy Group
    const avatarGroup = new THREE.Group();
    avatarGroup.position.y = -0.8;

    const skinMat = new THREE.MeshStandardMaterial({
      color: 0xe0ac69,
      roughness: 0.6,
      metalness: 0.1,
    });

    // Head
    const headGeo = new THREE.SphereGeometry(0.18, 32, 32);
    const head = new THREE.Mesh(headGeo, skinMat);
    head.position.y = 1.9;
    head.castShadow = true;
    avatarGroup.add(head);

    // Hair
    const hairGeo = new THREE.SphereGeometry(0.20, 24, 24, 0, Math.PI * 2, 0, Math.PI / 1.7);
    const hairMat = new THREE.MeshStandardMaterial({ color: 0x221814, roughness: 0.9 });
    const hair = new THREE.Mesh(hairGeo, hairMat);
    hair.position.set(0, 1.95, -0.02);
    avatarGroup.add(hair);

    // Neck
    const neckGeo = new THREE.CylinderGeometry(0.06, 0.08, 0.12, 16);
    const neck = new THREE.Mesh(neckGeo, skinMat);
    neck.position.y = 1.76;
    avatarGroup.add(neck);

    // Default Materials for Clothes
    const topMat = new THREE.MeshStandardMaterial({ color: 0x2d3436, roughness: 0.7 });
    const bottomMat = new THREE.MeshStandardMaterial({ color: 0x1e272e, roughness: 0.6 });
    const shoeMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5 });
    const accMat = new THREE.MeshStandardMaterial({ color: 0xf5a623, metalness: 0.8, roughness: 0.2 });

    // Torso / Top
    const torsoGeo = new THREE.CylinderGeometry(0.24, 0.19, 0.55, 24);
    const torso = new THREE.Mesh(torsoGeo, topMat);
    torso.position.y = 1.42;
    torso.castShadow = true;
    avatarGroup.add(torso);

    // Arms
    const armGeo = new THREE.CylinderGeometry(0.05, 0.045, 0.50, 16);
    const leftArm = new THREE.Mesh(armGeo, topMat);
    leftArm.position.set(-0.32, 1.40, 0);
    leftArm.rotation.z = 0.15;
    avatarGroup.add(leftArm);

    const rightArm = new THREE.Mesh(armGeo, topMat);
    rightArm.position.set(0.32, 1.40, 0);
    rightArm.rotation.z = -0.15;
    avatarGroup.add(rightArm);

    // Legs / Bottom
    const legGeo = new THREE.CylinderGeometry(0.08, 0.065, 0.75, 16);
    const leftLeg = new THREE.Mesh(legGeo, bottomMat);
    leftLeg.position.set(-0.13, 0.78, 0);
    leftLeg.castShadow = true;
    avatarGroup.add(leftLeg);

    const rightLeg = new THREE.Mesh(legGeo, bottomMat);
    rightLeg.position.set(0.13, 0.78, 0);
    rightLeg.castShadow = true;
    avatarGroup.add(rightLeg);

    // Shoes
    const shoeGeo = new THREE.BoxGeometry(0.12, 0.08, 0.22);
    const leftShoe = new THREE.Mesh(shoeGeo, shoeMat);
    leftShoe.position.set(-0.13, 0.36, 0.04);
    leftShoe.castShadow = true;
    avatarGroup.add(leftShoe);

    const rightShoe = new THREE.Mesh(shoeGeo, shoeMat);
    rightShoe.position.set(0.13, 0.36, 0.04);
    rightShoe.castShadow = true;
    avatarGroup.add(rightShoe);

    // Accessory (e.g. Necklace / Bag)
    const accGeo = new THREE.TorusGeometry(0.10, 0.015, 12, 24);
    const acc = new THREE.Mesh(accGeo, accMat);
    acc.position.set(0, 1.68, 0.12);
    acc.rotation.x = Math.PI / 3;
    acc.visible = false;
    avatarGroup.add(acc);

    scene.add(avatarGroup);

    meshesRef.current = {
      avatarGroup,
      torso,
      leftArm,
      rightArm,
      leftLeg,
      rightLeg,
      leftShoe,
      rightShoe,
      acc,
      topMat,
      bottomMat,
      shoeMat,
      accMat,
    };

    // 7. Mouse Orbit Drag interaction
    const onMouseDown = (e) => {
      isDraggingRef.current = true;
      prevMouseRef.current = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e) => {
      if (!isDraggingRef.current) return;
      const deltaX = e.clientX - prevMouseRef.current.x;
      avatarGroup.rotation.y += deltaX * 0.01;
      prevMouseRef.current = { x: e.clientX, y: e.clientY };
    };

    const onMouseUp = () => {
      isDraggingRef.current = false;
    };

    const dom = renderer.domElement;
    dom.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);

    // Touch support for mobile
    const onTouchStart = (e) => {
      if (e.touches.length === 1) {
        isDraggingRef.current = true;
        prevMouseRef.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      }
    };
    const onTouchMove = (e) => {
      if (!isDraggingRef.current || e.touches.length !== 1) return;
      const deltaX = e.touches[0].clientX - prevMouseRef.current.x;
      avatarGroup.rotation.y += deltaX * 0.015;
      prevMouseRef.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    };
    const onTouchEnd = () => { isDraggingRef.current = false; };

    dom.addEventListener('touchstart', onTouchStart);
    window.addEventListener('touchmove', onTouchMove);
    window.addEventListener('touchend', onTouchEnd);

    // 8. Animation loop
    let reqId;
    let clock = new THREE.Clock();

    const animate = () => {
      reqId = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();

      // Gentle breathing idle animation
      if (avatarGroup) {
        avatarGroup.position.y = -0.8 + Math.sin(elapsed * 1.5) * 0.008;
      }
      ring.rotation.z += 0.005;

      renderer.render(scene, camera);
    };
    animate();

    // 9. Resize handler
    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(reqId);
      window.removeEventListener('resize', handleResize);
      dom.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      dom.removeEventListener('touchstart', onTouchStart);
      window.removeEventListener('touchmove', onTouchMove);
      window.removeEventListener('touchend', onTouchEnd);
      renderer.dispose();
    };
  }, []);

  // Update clothing materials whenever outfit changes
  useEffect(() => {
    const meshes = meshesRef.current;
    if (!meshes || !meshes.avatarGroup) return;

    let hasDress = false;
    let hasTop = false;
    let hasBottom = false;
    let hasShoes = false;
    let hasAcc = false;

    outfit.forEach((item) => {
      const colorVal = COLOR_MAP[item.color] || 0x4a4a4a;

      if (item.category === 'dress') {
        hasDress = true;
        meshes.topMat.color.setHex(colorVal);
        meshes.bottomMat.color.setHex(colorVal);
      } else if (item.category === 'top') {
        hasTop = true;
        meshes.topMat.color.setHex(colorVal);
      } else if (item.category === 'bottom') {
        hasBottom = true;
        meshes.bottomMat.color.setHex(colorVal);
      } else if (item.category === 'shoes') {
        hasShoes = true;
        meshes.shoeMat.color.setHex(colorVal);
      } else if (item.category === 'accessory') {
        hasAcc = true;
        meshes.accMat.color.setHex(colorVal);
        meshes.acc.visible = true;
      }
    });

    if (!hasAcc && meshes.acc) {
      meshes.acc.visible = false;
    }
  }, [outfit]);

  const handleEnterVR = () => {
    if (vrSupported) {
      setVrStatusMsg('Entering WebXR Immersive Session...');
      if (onToggleVR) onToggleVR(true);
    } else {
      setVrStatusMsg('WebXR VR headset not detected. Operating in High-Fidelity 3D Desktop mode.');
      setTimeout(() => setVrStatusMsg(''), 4000);
    }
  };

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: '480px' }}>
      <div ref={mountRef} style={{ width: '100%', height: '100%', borderRadius: '14px', overflow: 'hidden' }} />
      
      {/* 3D Controls Overlay */}
      <div style={{
        position: 'absolute',
        bottom: '16px',
        left: '16px',
        right: '16px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        background: 'rgba(18, 20, 31, 0.75)',
        backdropFilter: 'blur(10px)',
        padding: '8px 16px',
        borderRadius: '9999px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        pointerEvents: 'auto',
      }}>
        <div style={{ fontSize: '0.8rem', color: '#9aa0b8' }}>
          🖱️ Click & Drag to rotate 3D Avatar
        </div>
        <button
          onClick={handleEnterVR}
          className="btn-secondary"
          style={{ padding: '0.35rem 0.9rem', fontSize: '0.8rem' }}
        >
          🥽 {vrSupported ? 'Enter VR Mode' : 'VR View (3D Fallback)'}
        </button>
      </div>

      {vrStatusMsg && (
        <div style={{
          position: 'absolute',
          top: '16px',
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'rgba(108, 99, 255, 0.9)',
          color: '#fff',
          padding: '8px 18px',
          borderRadius: '9999px',
          fontSize: '0.85rem',
          fontWeight: 600,
          boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
          zIndex: 10,
        }}>
          {vrStatusMsg}
        </div>
      )}
    </div>
  );
}
