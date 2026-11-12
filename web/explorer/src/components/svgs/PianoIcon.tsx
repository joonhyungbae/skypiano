export default function PianoIcon() {
  const gradientId = 'pianopus-mark-grad';
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 32 32"
      className="pianopus-mark"
      aria-hidden="true"
      style={{ display: 'block', width: 32, height: 32 }}
    >
      <defs>
        <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style={{ stopColor: '#0F0F70' }} />
          <stop offset="100%" style={{ stopColor: '#4B1E78' }} />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="6" fill={`url(#${gradientId})`} />
      <g fill="white">
        <rect x="6" y="10" width="4" height="12" rx="1" />
        <rect x="11" y="10" width="4" height="12" rx="1" />
        <rect x="16" y="10" width="4" height="12" rx="1" />
        <rect x="21" y="10" width="4" height="12" rx="1" />
        <rect x="9" y="10" width="2.5" height="7" rx="0.5" fill="#1a1a2e" />
        <rect x="14" y="10" width="2.5" height="7" rx="0.5" fill="#1a1a2e" />
        <rect x="19.5" y="10" width="2.5" height="7" rx="0.5" fill="#1a1a2e" />
      </g>
    </svg>
  );
}
