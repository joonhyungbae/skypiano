import { Outlet } from 'react-router-dom';
import { styled } from '@mui/material/styles';
import { TrialsMenuProvider } from './components/common/TrialsMenu';
import DesktopOnlyGate from './components/common/DesktopOnlyGate';

const Shell = styled('div')({
  height: '100vh',
  display: 'flex',
  flexDirection: 'column',
  background: 'transparent',
  overflow: 'hidden',
});

const Main = styled('main')({
  flex: '1 1 0',
  minHeight: 0,
  display: 'flex',
  flexDirection: 'column',
});

export default function App() {
  return (
    <DesktopOnlyGate>
      <TrialsMenuProvider>
        <Shell>
          <Main>
            <Outlet />
          </Main>
        </Shell>
      </TrialsMenuProvider>
    </DesktopOnlyGate>
  );
}
