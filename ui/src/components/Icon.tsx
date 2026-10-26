import type { SVGProps } from "react";

type IconName = "cube" | "play" | "chart" | "flask" | "agent" | "report" | "bell" | "plus";

const paths: Record<IconName, string> = {
  cube: "M4 6.5 12 2l8 4.5v9L12 20l-8-4.5v-9Zm8 4.5 8-4.5M12 11 4 6.5M12 11v9",
  play: "M7 4.5v15l12-7.5L7 4.5Z",
  chart: "M4 19V5m0 14h16M7 15l3-4 3 2 5-7",
  flask: "M9 3h6m-5 0v6l-5 9a2 2 0 0 0 1.8 3h10.4A2 2 0 0 0 19 18l-5-9V3M7 16h10",
  agent: "M12 3 10.8 7.8 6 9l4.8 1.2L12 15l1.2-4.8L18 9l-4.8-1.2L12 3Zm-6 12-.6 2.4L3 18l2.4.6L6 21l.6-2.4L9 18l-2.4-.6L6 15Z",
  report: "M6 3h9l3 3v15H6V3Zm9 0v4h4M9 11h6M9 15h6",
  bell: "M18 9a6 6 0 0 0-12 0c0 7-3 7-3 8h18c0-1-3-1-3-8Zm-8 11h4",
  plus: "M12 5v14M5 12h14",
};

export function Icon({ name, ...props }: { name: IconName } & SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d={paths[name]} />
    </svg>
  );
}
