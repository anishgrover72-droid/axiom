import { Page } from "../content";

const LINKS: { page: Page; label: string }[] = [
  { page: "home", label: "Home" },
  { page: "pipeline", label: "Pipeline" },
  { page: "tech", label: "Tech" },
];

export function Nav({ page, onNavigate }: { page: Page; onNavigate: (p: Page) => void }) {
  return (
    <nav className="nav">
      <button className="nav-brand" onClick={() => onNavigate("home")}>
        <svg viewBox="0 0 40 40" fill="none" aria-hidden="true">
          <circle cx="20" cy="20" r="9" fill="#ff7a45" opacity="0.16" />
          <circle cx="20" cy="20" r="9" stroke="#ff7a45" strokeWidth="1.5" />
          <ellipse
            cx="20"
            cy="20"
            rx="17"
            ry="6.5"
            stroke="#796e63"
            strokeWidth="1.2"
            transform="rotate(-28 20 20)"
          />
          <circle cx="33.5" cy="13" r="2.2" fill="#ff7a45" />
        </svg>
        <span>AXIOM</span>
      </button>
      <div className="nav-links">
        {LINKS.map((l) => (
          <button
            key={l.page}
            className={`nav-link${page === l.page ? " active" : ""}`}
            onClick={() => onNavigate(l.page)}
          >
            {l.label}
          </button>
        ))}
        <button
          className={`nav-cta${page === "console" ? " active" : ""}`}
          onClick={() => onNavigate("console")}
        >
          Launch console
        </button>
      </div>
    </nav>
  );
}
