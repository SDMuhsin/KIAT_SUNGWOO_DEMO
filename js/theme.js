// theme.js - shared palette + material factories. Colour-blind-safe (Okabe-Ito-ish) on a dark background.
export const BG = 0x0b0f17;
/** Numeric hex colours (for THREE) - semantic keys, use these rather than raw hex. */
export const COLORS = {
  weights: 0x4a90e2,      // blue
  activations: 0x3ab795,  // green
  gradients: 0xf08a24,    // orange
  optimizer: 0x9b6bd6,    // purple
  physics: 0xe0408a,      // red/magenta
  sage: 0x2fd0e6,         // cyan
  sageAccent: 0xf2c14e,   // gold
  geometry: 0x8a94a6,     // neutral steel
  points: 0xd8dee9,
  text: 0xe8edf5, muted: 0x8793a6, grid: 0x1c2433,
  memoryBound: 0xe0408a, computeBound: 0x2fd0e6,
};
/** CSS strings of the same palette (HUD / sprite labels). */
export const CSS = Object.fromEntries(Object.entries(COLORS).map(([k, v]) => [k, '#' + v.toString(16).padStart(6, '0')]));

/** Standard material: material(THREE,'weights',{opacity,emissive,wire}). Transparent so util.gateGroup can fade it. */
export function material(THREE, key, { opacity = 1, emissive = 0.15, wire = false } = {}) {
  const c = COLORS[key] ?? key;
  const m = new THREE.MeshStandardMaterial({ color: c, emissive: c, emissiveIntensity: emissive, roughness: 0.55, metalness: 0.1, transparent: true, opacity, wireframe: wire });
  m.userData.baseOpacity = opacity;
  return m;
}
/** Unlit line/point material (cheap). */
export function lineMaterial(THREE, key, opacity = 1) {
  const m = new THREE.LineBasicMaterial({ color: COLORS[key] ?? key, transparent: true, opacity });
  m.userData.baseOpacity = opacity; return m;
}
