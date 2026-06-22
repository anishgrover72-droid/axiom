import { CSSProperties, ReactNode, useEffect, useRef, useState } from "react";

// Reveals its children with a fade-up the first time they scroll into view.
export function Reveal({
  children,
  delay = 0,
  className = "",
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setShown(true);
          io.disconnect();
        }
      },
      { threshold: 0.15 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const style: CSSProperties = { transitionDelay: `${delay}s` };
  return (
    <div ref={ref} className={`reveal${shown ? " in" : ""} ${className}`.trim()} style={style}>
      {children}
    </div>
  );
}
