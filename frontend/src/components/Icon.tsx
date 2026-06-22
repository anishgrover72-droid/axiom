// Minimal line-icon set for the pipeline flowchart. Stroke inherits currentColor.
const PATHS: Record<string, JSX.Element> = {
  database: (
    <>
      <ellipse cx="12" cy="5" rx="7" ry="3" />
      <path d="M5 5v6c0 1.7 3.1 3 7 3s7-1.3 7-3V5" />
      <path d="M5 11v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6" />
    </>
  ),
  sparkles: (
    <>
      <path d="M12 3l1.7 4.3L18 9l-4.3 1.7L12 15l-1.7-4.3L6 9l4.3-1.7z" />
      <path d="M18.5 14l.8 1.9 1.9.8-1.9.8-.8 1.9-.8-1.9-1.9-.8 1.9-.8z" />
    </>
  ),
  scissors: (
    <>
      <circle cx="6" cy="6" r="2.6" />
      <circle cx="6" cy="18" r="2.6" />
      <path d="M20 4L8.2 15.8" />
      <path d="M14.6 14.6L20 20" />
      <path d="M8.2 8.2L11.5 11.5" />
    </>
  ),
  sliders: (
    <>
      <path d="M4 7h16" />
      <path d="M4 12h16" />
      <path d="M4 17h16" />
      <circle cx="9" cy="7" r="2" />
      <circle cx="15" cy="12" r="2" />
      <circle cx="8" cy="17" r="2" />
    </>
  ),
  beaker: (
    <>
      <path d="M9 3h6" />
      <path d="M10 3v6l-5 9.5a1 1 0 00.9 1.5h12.2a1 1 0 00.9-1.5L14 9V3" />
      <path d="M7.5 14h9" />
    </>
  ),
  layers: (
    <>
      <path d="M12 3l9 5-9 5-9-5 9-5z" />
      <path d="M3 13l9 5 9-5" />
    </>
  ),
  repeat: (
    <>
      <path d="M17 2l3 3-3 3" />
      <path d="M20 5H9a4 4 0 00-4 4v1" />
      <path d="M7 22l-3-3 3-3" />
      <path d="M4 19h11a4 4 0 004-4v-1" />
    </>
  ),
  gauge: (
    <>
      <path d="M3.5 18a9 9 0 1117 0" />
      <path d="M12 14l4-3" />
      <circle cx="12" cy="14.5" r="1.3" />
    </>
  ),
  broadcast: (
    <>
      <circle cx="12" cy="12" r="2" />
      <path d="M16.2 7.8a6 6 0 010 8.4" />
      <path d="M7.8 16.2a6 6 0 010-8.4" />
      <path d="M19 5a10 10 0 010 14" />
      <path d="M5 19A10 10 0 015 5" />
    </>
  ),
};

export function Icon({ name }: { name: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {PATHS[name] ?? PATHS.layers}
    </svg>
  );
}
