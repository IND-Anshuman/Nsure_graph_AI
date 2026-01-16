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
        border: "hsl(222 47% 12%)",
        input: "hsl(222 47% 12%)",
        ring: "hsl(28 35% 55%)",
        background: "hsl(225 30% 3%)",
        foreground: "hsl(215 25% 90%)",
        primary: {
          DEFAULT: "hsl(215 25% 90%)",
          foreground: "hsl(225 30% 3%)",
        },
        secondary: {
          DEFAULT: "hsl(225 30% 7%)",
          foreground: "hsl(215 25% 85%)",
        },
        accent: {
          DEFAULT: "hsl(35 60% 55%)",
          foreground: "hsl(225 30% 3%)",
        },
        muted: {
          DEFAULT: "hsl(225 20% 8%)",
          foreground: "hsl(215 15% 65%)",
        },
        ivory: "#f8f9fa",
        charcoal: "#010204",
        gold: "#d4af37",
      },
      fontFamily: {
        serif: ["EB Garamond", "serif"],
        sans: ["Inter", "sans-serif"],
      },
      borderRadius: {
        lg: "0.5rem",
        md: "0.25rem",
        sm: "0.125rem",
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
