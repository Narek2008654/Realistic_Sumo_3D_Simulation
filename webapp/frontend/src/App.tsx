import { Navigate, Route, Routes } from 'react-router-dom';
import { AppShell } from './components/AppShell';
import Arena from './pages/Arena';
import Hardware from './pages/Hardware';
import Models from './pages/Models';
import Opponents from './pages/Opponents';
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
        <Route path="/arena" element={<Arena />} />
        <Route path="/opponents" element={<Opponents />} />
        <Route path="*" element={<Navigate to="/hardware" replace />} />
      </Routes>
    </AppShell>
  );
}
