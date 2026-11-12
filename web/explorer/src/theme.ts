import { createTheme } from '@mui/material/styles';

// Brand color tokens
export const BRAND = {
  PRIMARY: '#0F0F70',    // navy
  SECONDARY: '#004191',  // blue
  TERTIARY: '#4B1E78',   // purple
} as const;

// Shared surface and text tokens
export const SURFACE = {
  bg: '#F7F7FA',
  paper: '#FFFFFF',
  border: 'rgba(15, 15, 112, 0.12)',
  borderStrong: 'rgba(15, 15, 112, 0.24)',
  textPrimary: '#1A1A2E',
  textSecondary: '#5A5A6E',
  textMuted: '#8A8A9A',
  shadow: '0 4px 16px rgba(15, 15, 112, 0.08)',
  shadowHover: '0 8px 24px rgba(15, 15, 112, 0.14)',
} as const;

// Signature gradients
export const GRADIENT = {
  brand: `linear-gradient(135deg, ${BRAND.PRIMARY}, ${BRAND.TERTIARY})`,
  cool: `linear-gradient(135deg, ${BRAND.PRIMARY}, ${BRAND.SECONDARY})`,
} as const;

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: BRAND.PRIMARY, contrastText: '#FFFFFF' },
    secondary: { main: BRAND.SECONDARY, contrastText: '#FFFFFF' },
    info: { main: BRAND.TERTIARY, contrastText: '#FFFFFF' },
    error: { main: '#D32F2F' },
    success: { main: '#2E7D32' },
    warning: { main: '#ED6C02' },
    background: {
      default: SURFACE.bg,
      paper: SURFACE.paper,
    },
    text: {
      primary: SURFACE.textPrimary,
      secondary: SURFACE.textSecondary,
    },
    divider: SURFACE.border,
  },
  shape: { borderRadius: 10 },
  typography: {
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      '"Apple SD Gothic Neo"',
      '"Malgun Gothic"',
      'sans-serif',
    ].join(','),
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: { background: SURFACE.bg },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: SURFACE.paper,
          border: `1px solid ${SURFACE.border}`,
          boxShadow: SURFACE.shadow,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: { textTransform: 'none', fontWeight: 600 },
      },
    },
  },
});

export default theme;
