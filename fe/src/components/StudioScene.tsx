export function StudioScene() {
  return (
    <svg
      className="studio-scene"
      viewBox="0 0 800 430"
      fill="none"
      role="img"
      aria-label="Minh họa người ngồi làm việc với các điểm khung xương, không phải kết quả nhận diện"
    >
      <defs>
        <linearGradient id="room" x2="1" y2="1">
          <stop stopColor="#edf3f8" />
          <stop offset="1" stopColor="#dee7f1" />
        </linearGradient>
        <linearGradient id="shirt" x2="1" y2="1">
          <stop stopColor="#869ade" />
          <stop offset="1" stopColor="#677cc1" />
        </linearGradient>
        <pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse">
          <circle cx="1" cy="1" r=".8" fill="#98abc4" opacity=".3" />
        </pattern>
      </defs>
      <rect width="800" height="430" fill="url(#room)" />
      <rect width="800" height="430" fill="url(#dots)" />
      <path d="M0 338H800V430H0" fill="#d5e0ee" />
      <path d="M0 338H800M200 430 320 338M550 430 500 338" stroke="#c9d6e7" />
      <rect x="75" y="60" width="168" height="194" rx="5" fill="#cad8e8" />
      <rect x="84" y="69" width="150" height="176" rx="2" fill="#f6f9fc" />
      <path d="M84 175 152 119 234 184V245H84" fill="#e5eef7" />
      <path d="M84 220 173 156 234 202V245H84" fill="#dbe7f1" />
      <path d="M159 69V245M84 154H234" stroke="#cad8e8" strokeWidth="7" />
      <rect x="564" y="82" width="130" height="7" rx="3" fill="#bdcce0" />
      <rect x="583" y="55" width="9" height="27" rx="2" fill="#afbed4" />
      <rect x="596" y="46" width="12" height="36" rx="2" fill="#8fa5c7" />
      <path d="m615 52 8-2 8 31-8 2" fill="#a0b8cc" />
      <ellipse cx="405" cy="368" rx="180" ry="20" fill="#bacadd" opacity=".5" />
      <path d="M614 342v-51" stroke="#7eaa9e" strokeWidth="4" />
      <path
        d="M614 306c-31 2-42-28-26-39 24 2 30 19 26 39M615 315c29-4 42-26 27-38-26 1-31 22-27 38M614 289c-18-16-13-47 4-48 19 14 10 37-4 48"
        fill="#8fb9ab"
      />
      <path d="M588 325h53l-8 40h-37z" fill="#eff3f7" />
      <path d="M588 325h53" stroke="#fff" strokeWidth="5" />
      <path
        d="M316 246v91M316 337l-33 24M316 337l33 24"
        stroke="#899bb5"
        strokeWidth="9"
        strokeLinecap="round"
      />
      <rect x="274" y="170" width="25" height="100" rx="12" fill="#a9b9d2" />
      <rect x="287" y="253" width="100" height="18" rx="9" fill="#97abc9" />
      <path
        d="M345 242 405 259 395 330"
        stroke="#435578"
        strokeWidth="32"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <path
        d="M324 242 371 277 361 340"
        stroke="#52678d"
        strokeWidth="30"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <path
        d="M352 338h22l19 15q4 10-8 10h-35zM387 329h22l17 13q8 11-6 13h-36z"
        fill="#f8fafc"
      />
      <path
        d="M327 144q-24 3-22 33l5 62q24 21 51 9l-9-75q-4-28-25-29"
        fill="url(#shirt)"
      />
      <path
        d="M333 155v-24"
        stroke="#e2b59c"
        strokeWidth="17"
        strokeLinecap="round"
      />
      <path
        d="M317 114q-2-26 19-27 27-1 26 29l-4 20q-13 13-31-2z"
        fill="#e9c1aa"
      />
      <path
        d="M315 116q-13-31 13-37 35-6 35 28l-20-4-6 16-8-6-4 13z"
        fill="#3c4b69"
      />
      <path
        d="M337 172 365 212 413 218"
        stroke="#8195d5"
        strokeWidth="25"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M403 218h26"
        stroke="#e9c1aa"
        strokeWidth="13"
        strokeLinecap="round"
      />
      <path
        d="M318 175 332 219 380 228"
        stroke="#91a3df"
        strokeWidth="22"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M376 227h28"
        stroke="#edc6ad"
        strokeWidth="12"
        strokeLinecap="round"
      />
      <path d="M397 240v114M559 240v114" stroke="#a0aec5" strokeWidth="10" />
      <rect x="375" y="232" width="205" height="13" rx="5" fill="#f8fafc" />
      <path d="M435 225h89l17-66h-85z" fill="#9baecb" />
      <path d="M442 218h75l13-53h-75z" fill="#c7d5e8" />
      <rect x="415" y="225" width="109" height="6" rx="3" fill="#8d9fbc" />
      <path d="M548 207v22h17v-22z" fill="#a5c0c4" />
      <path
        d="M565 211q14 0 8 11h-8"
        fill="none"
        stroke="#a5c0c4"
        strokeWidth="4"
      />
      <g stroke="#18bca2" strokeWidth="2.4" fill="none" opacity=".94">
        <path d="M339 110 329 160 345 242 399 260 394 332M329 160 336 180 365 213 414 219M329 160 317 177 332 218 387 228M317 177 323 242 369 278 361 339M323 242 345 242" />
      </g>
      <g fill="#effffc" stroke="#19b79e" strokeWidth="2.5">
        {[
          [339, 110],
          [329, 160],
          [336, 180],
          [317, 177],
          [365, 213],
          [414, 219],
          [332, 218],
          [387, 228],
          [323, 242],
          [345, 242],
          [399, 260],
          [369, 278],
          [394, 332],
          [361, 339],
        ].map(([x, y]) => (
          <circle key={`${x}-${y}`} cx={x} cy={y} r="4.5" />
        ))}
      </g>
      <g fill="none" stroke="#7893bf" strokeWidth="2" opacity=".5">
        <path d="M278 101V77h24M393 77h24v24M278 334v25h24M417 334v25h-24" />
      </g>
      <path
        d="M245 180h-25m234 109h35"
        stroke="#99b3c7"
        strokeDasharray="3 4"
      />
      <circle cx="215" cy="180" r="4" fill="#99b3c7" />
      <circle cx="494" cy="289" r="4" fill="#99b3c7" />
    </svg>
  );
}
