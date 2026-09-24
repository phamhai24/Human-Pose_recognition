const poses = [
  "M48 34 L49 64 L77 66 L74 96 M49 64 L37 80 L39 99 M48 39 L62 53 L81 51 M47 40 L35 57 L59 58",
  "M37 34 L48 64 L75 66 L78 96 M48 64 L39 83 L40 99 M38 39 L50 55 L70 59 M37 39 L28 58 L45 62",
  "M27 63 L53 65 L77 59 L99 63 M53 65 L76 72 L100 73 M30 64 L38 77 L53 73 M30 61 L38 48 L46 58",
  "M40 34 L43 66 L66 64 L91 50 M43 66 L68 76 L90 71 M40 40 L52 55 L67 51 M40 40 L27 57 L41 65",
  "M58 32 L56 64 L45 98 M56 64 L69 98 M57 39 L40 53 L34 70 M58 39 L74 51 L80 69",
  "M59 32 L53 62 L30 89 M53 62 L75 80 L88 97 M57 38 L39 51 L27 48 M57 38 L73 52 L79 43",
];
export function PoseFigure({
  pose = 0,
  color = "#3563e9",
}: {
  pose?: number;
  color?: string;
}) {
  const head =
    pose === 2
      ? [17, 62]
      : pose === 1
        ? [33, 23]
        : pose === 3
          ? [40, 23]
          : pose === 0
            ? [48, 23]
            : [59, 21];
  return (
    <svg viewBox="0 0 120 112" fill="none" aria-hidden="true">
      <ellipse cx="60" cy="104" rx="37" ry="4" fill={color} opacity=".08" />
      {(pose < 2 || pose === 3) && (
        <path
          d="M27 45v30h41M33 75v25M64 75v25"
          stroke={color}
          strokeWidth="3"
          opacity=".2"
          strokeLinecap="round"
        />
      )}
      <path
        d={poses[pose]}
        stroke={color}
        strokeWidth="6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx={head[0]} cy={head[1]} r="8" fill={color} />
    </svg>
  );
}
