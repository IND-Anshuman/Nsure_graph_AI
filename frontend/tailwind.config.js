/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        border: "hsl(217 19% 27%)",
        input: "hsl(217 19% 27%)",
        ring: "hsl(171 75% 55%)",
        background: "hsl(222 47% 11%)",
        foreground: "hsl(210 40% 98%)",
        primary: {
          DEFAULT: "hsl(171 75% 55%)",
          foreground: "hsl(222 47% 11%)",
        },
        secondary: {
          DEFAULT: "hsl(217 32% 17%)",
          foreground: "hsl(210 40% 98%)",
        },
        accent: {
          DEFAULT: "hsl(171 75% 55%)",
          foreground: "hsl(222 47% 11%)",
        },
        muted: {
          DEFAULT: "hsl(217 32% 17%)",
          foreground: "hsl(215 20% 65%)",
        },
      },
      borderRadius: {
        lg: "0.75rem",
        md: "0.5rem",
        sm: "0.25rem",
      },
      keyframes: {
        "float": {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-20px)" },
        },
        "fade-in": {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "float": "float 3s ease-in-out infinite",
        "fade-in": "fade-in 0.5s ease-out",
      },
    },
  },
  plugins: [],
}
