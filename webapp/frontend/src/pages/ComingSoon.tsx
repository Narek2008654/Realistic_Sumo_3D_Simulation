// On-brand "coming soon" stub for routes not yet built (Train / Arena /
// Opponents) so navigation works end to end.
import { Panel, Reveal } from '../components/ui';

export default function ComingSoon({
  section,
  blurb,
}: {
  section: string;
  blurb: string;
}) {
  return (
    <Reveal>
      <Panel title={section} live ticks className="max-w-2xl">
        <div className="flex flex-col items-start gap-3 py-6">
          <span
            className="micro"
            style={{ color: 'var(--accent)', letterSpacing: '.18em' }}
          >
            ◇ MODULE OFFLINE
          </span>
          <h3 className="font-display text-[28px] font-semibold uppercase tracking-wide">
            {section}
          </h3>
          <p className="num max-w-md text-fg-1" style={{ fontSize: 13 }}>
            {blurb}
          </p>
          <div
            className="mt-2 h-px w-full"
            style={{
              background:
                'linear-gradient(90deg, var(--accent-dim), transparent)',
            }}
          />
          <span className="micro text-fg-2" style={{ fontSize: 10 }}>
            COMING SOON · WIRED IN A LATER FEATURE DROP
          </span>
        </div>
      </Panel>
    </Reveal>
  );
}
