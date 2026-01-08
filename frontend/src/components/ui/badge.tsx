import * as React from "react";
import { cn } from "@/lib/utils";

const Badge = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & {
    variant?: "default" | "success" | "outline";
  }
>(({ className, variant = "default", ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn(
        "inline-flex items-center rounded-full px-3 py-1 text-xs font-medium transition-colors",
        {
          "bg-primary/20 text-primary border border-primary/30": variant === "default",
          "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30": variant === "success",
          "border border-border bg-transparent": variant === "outline",
        },
        className
      )}
      {...props}
    />
  );
});
Badge.displayName = "Badge";

export { Badge };
