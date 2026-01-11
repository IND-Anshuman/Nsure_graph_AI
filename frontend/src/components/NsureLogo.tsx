import nsureIconUrl from "@/assets/nsure-icon.png";
import { cn } from "@/lib/utils";

export function NsureLogo({
  className,
  alt = "Nsure AI",
}: {
  className?: string;
  alt?: string;
}) {
  return (
    <img
      src={nsureIconUrl}
      alt={alt}
      className={cn("block object-contain", className)}
      draggable={false}
    />
  );
}
