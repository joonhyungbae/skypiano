import type { ReactNode } from 'react';
import { Link, useMatch } from 'react-router-dom';
import { styled } from '@mui/material/styles';
import HomeIcon from '@mui/icons-material/Home';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import { TrialsMenuToggle } from './TrialsMenu';
import { BRAND, SURFACE } from '@/theme';

const Root = styled('header')({
  display: 'flex',
  alignItems: 'center',
  gap: 10,
  height: 52,
  padding: '0 16px',
  background: SURFACE.paper,
  borderBottom: `1px solid ${SURFACE.border}`,
  boxShadow: '0 4px 14px rgba(15, 15, 112, 0.05)',
  flexShrink: 0,
  zIndex: 10,
  overflow: 'hidden',
});

const RightSlot = styled('div')({
  marginLeft: 'auto',
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  flexShrink: 0,
});

/** The explorer is mounted under the project page, which owns the base above it. */
const PROJECT_PAGE_URL = import.meta.env.BASE_URL.replace(/explorer\/?$/, '');

const ProjectPageLink = styled('a')({
  display: 'inline-flex',
  alignItems: 'center',
  gap: 5,
  height: 32,
  padding: '0 10px',
  borderRadius: 8,
  border: `1px solid ${SURFACE.border}`,
  background: SURFACE.paper,
  color: SURFACE.textSecondary,
  fontSize: 12,
  fontWeight: 600,
  textDecoration: 'none',
  whiteSpace: 'nowrap',
  transition: 'all 0.15s ease',
  flexShrink: 0,
  '&:hover': {
    color: BRAND.PRIMARY,
    borderColor: SURFACE.borderStrong,
    background: `${BRAND.PRIMARY}08`,
  },
  '& svg': { fontSize: 16 },
});

const HomeLink = styled(Link)({
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  width: 32,
  height: 32,
  borderRadius: 8,
  border: `1px solid ${SURFACE.border}`,
  background: SURFACE.paper,
  color: SURFACE.textSecondary,
  textDecoration: 'none',
  transition: 'all 0.15s ease',
  flexShrink: 0,
  '&:hover': {
    color: BRAND.PRIMARY,
    borderColor: SURFACE.borderStrong,
    background: `${BRAND.PRIMARY}08`,
  },
  '& svg': { fontSize: 18 },
});

/**
 * Full-width top chrome. Pages supply their own context (brand, trial title) via
 * children; the home button + TrialsMenu hamburger sit on the right.
 */
export default function TopBar({ children }: { children?: ReactNode }) {
  const onHome = useMatch('/');
  return (
    <Root>
      {children}
      <RightSlot>
        <ProjectPageLink href={PROJECT_PAGE_URL} title="Back to the project page">
          <ChevronLeftIcon />
          Project page
        </ProjectPageLink>
        {!onHome && (
          <HomeLink to="/" aria-label="Go to home" title="Home">
            <HomeIcon />
          </HomeLink>
        )}
        <TrialsMenuToggle />
      </RightSlot>
    </Root>
  );
}
