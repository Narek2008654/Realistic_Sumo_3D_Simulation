import { Navigate, Route, Routes } from 'react-router-dom';
import { AppShell } from './components/AppShell';
import ComingSoon from './pages/ComingSoon';
import Hardware from './pages/Hardware';
import Models from './pages/Models';
import Robots from './pages/Robots';
import Train from './pages/Train';

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<Navigate to="/hardware" replace />} />
        <Route path="/hardware" element={<Hardware />} />
        <Route path="/robots" element={<Robots />} />
        <Route path="/models" element={<Models />} />
        <Route path="/train" element={<Train />} />
        <Route
          path="/arena"
          element={
            <ComingSoon
              section="Arena"
              blurb="Stage head-to-head battles between checkpoints and the opponent zoo, scrub the dohyo-cam replay, and read win / self-out / push statistics."
            />
          }
        />
        <Route
          path="/opponents"
          element={
            <ComingSoon
              section="Opponents"
              blurb="Build rule-DSL opponents and bind them to a hardware spec — dodger, spinner, rammer, wedger and friends, ready to drop into the arena."
            />
          }
        />
        <Route path="*" element={<Navigate to="/hardware" replace />} />
      </Routes>
    </AppShell>
  );
}
