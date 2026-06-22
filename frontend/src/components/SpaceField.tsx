import { CSSProperties, useMemo } from "react";

// Persistent two-layer starfield behind every page. The layers parallax at different
// rates off the cursor (--px / --py set on the app root) for depth.
interface Star {
  x: number;
  y: number;
  size: number;
  delay: number;
  dur: number;
}

function makeStars(n: number): Star[] {
  return Array.from({ length: n }, () => ({
    x: Math.random() * 100,
    y: Math.random() * 100,
    size: Math.random() * 1.6 + 0.4,
    delay: Math.random() * 6,
    dur: Math.random() * 4 + 3,
  }));
}

function starStyle(s: Star): CSSProperties {
  return {
    left: `${s.x}%`,
    top: `${s.y}%`,
    width: `${s.size}px`,
    height: `${s.size}px`,
    animationDelay: `${s.delay}s`,
    animationDuration: `${s.dur}s`,
  };
}

export function SpaceField() {
  const far = useMemo(() => makeStars(130), []);
  const near = useMemo(() => makeStars(60), []);
  return (
    <div className="space" aria-hidden="true">
      <div className="galaxy" />
      <div className="space-veil" />
      <div className="stars far">
        {far.map((s, i) => (
          <span key={i} style={starStyle(s)} />
        ))}
      </div>
      <div className="stars near">
        {near.map((s, i) => (
          <span key={i} style={starStyle(s)} />
        ))}
      </div>
    </div>
  );
}
