import { MouseEvent, useRef, useState } from "react";
import { Nav } from "./components/Nav";
import { SpaceField } from "./components/SpaceField";
import { Page } from "./content";
import { Console } from "./pages/Console";
import { Home } from "./pages/Home";
import { Pipeline } from "./pages/Pipeline";
import { Tech } from "./pages/Tech";

export default function App() {
  const [page, setPage] = useState<Page>("home");
  const rootRef = useRef<HTMLDivElement>(null);
  const frame = useRef<number>(0);

  function onMove(e: MouseEvent) {
    const px = (e.clientX / window.innerWidth) * 2 - 1;
    const py = (e.clientY / window.innerHeight) * 2 - 1;
    cancelAnimationFrame(frame.current);
    frame.current = requestAnimationFrame(() => {
      const el = rootRef.current;
      if (!el) return;
      el.style.setProperty("--px", px.toFixed(3));
      el.style.setProperty("--py", py.toFixed(3));
    });
  }

  function navigate(p: Page) {
    setPage(p);
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  return (
    <div className="app-root" ref={rootRef} onMouseMove={onMove}>
      <SpaceField />
      <Nav page={page} onNavigate={navigate} />
      <main className="page" key={page}>
        {page === "home" && <Home onNavigate={navigate} />}
        {page === "pipeline" && <Pipeline />}
        {page === "tech" && <Tech />}
        {page === "console" && <Console />}
      </main>
    </div>
  );
}
