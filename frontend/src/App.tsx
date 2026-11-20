import { Link, NavLink, Route, Routes, useSearchParams } from "react-router-dom";
import JobProgress from "./pages/JobProgress";
import ModelCard from "./pages/ModelCard";
import NewAssessment from "./pages/NewAssessment";
import ReportView from "./report/ReportView";

function Chrome() {
  const [q] = useSearchParams();
  if (q.get("print") === "1") return null; // the PDF carries the notice in its footer
  const link = ({ isActive }: { isActive: boolean }) => `px-2 py-1 ${isActive ? "font-semibold underline" : "hover:underline"}`;
  return (
    <header className="no-print sticky top-0 z-10">
      <div role="note" className="border-b border-[#e6d28a] bg-[#fff8e1] px-4 py-1.5 text-[13px] font-semibold">
        Research use only — not a diagnostic device
      </div>
      <nav aria-label="Main" className="flex items-center gap-3 border-b border-line bg-white px-4 py-1.5 text-sm">
        <Link to="/" className="mr-3 font-bold">Cognitive assessment</Link>
        <NavLink to="/" end className={link}>New assessment</NavLink>
        <NavLink to="/model-card" className={link}>Model card</NavLink>
      </nav>
    </header>
  );
}

export default function App() {
  return (
    <>
      <Chrome />
      <Routes>
        <Route path="/" element={<NewAssessment />} />
        <Route path="/jobs/:id" element={<JobProgress />} />
        <Route path="/report/:id" element={<ReportView />} />
        <Route path="/model-card" element={<ModelCard />} />
        <Route path="*" element={<NewAssessment />} />
      </Routes>
    </>
  );
}
