import { lazy, Suspense, useEffect } from 'react';
import { createBrowserRouter, useMatches } from 'react-router-dom';
import App from './App';

const BRAND_TITLE = 'SKY Piano';

const HomeView = lazy(() => import('./views/HomeView'));
const SampleView = lazy(() => import('./views/SampleView'));

function TitleUpdater() {
  const matches = useMatches();
  useEffect(() => {
    const last = matches[matches.length - 1];
    const handle = last?.handle as { title?: string } | undefined;
    // Always lead with the brand so a narrow tab keeps "SKY Piano" visible.
    document.title = handle?.title ? `${BRAND_TITLE} · ${handle.title}` : BRAND_TITLE;
  }, [matches]);
  return null;
}

function Shell() {
  return (
    <>
      <TitleUpdater />
      <Suspense fallback={null}>
        <App />
      </Suspense>
    </>
  );
}

export const router = createBrowserRouter(
  [
    {
      path: '/',
      element: <Shell />,
      children: [
        { index: true, element: <HomeView />, handle: { title: 'Home' } },
        {
          path: 'sample/:trialId',
          element: <SampleView />,
          handle: { title: 'Sample' },
        },
      ],
    },
  ],
  // The site is served from a GitHub Pages project path, so routes resolve
  // against the base Vite was built with rather than the domain root.
  { basename: import.meta.env.BASE_URL },
);
