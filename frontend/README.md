# Nsure AI Frontend

High-end production frontend for the Nsure AI GraphRAG system. A dark-themed, futuristic knowledge graph interface built with React, TypeScript, and modern web technologies.

## Architecture

```
frontend/
├── src/
│   ├── components/
│   │   ├── ui/              # Reusable UI primitives (shadcn/ui style)
│   │   ├── BackgroundParticles.tsx
│   │   └── FileUpload.tsx
│   ├── pages/
│   │   ├── LandingPage.tsx  # Product overview & hero
│   │   └── AgentPage.tsx    # Main interaction interface
│   ├── lib/
│   │   └── utils.ts         # Utility functions
│   ├── App.tsx              # Router configuration
│   ├── main.tsx             # Application entry
│   └── index.css            # Global styles
└── ...config files
```

## Tech Stack

- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **Routing**: React Router v6
- **UI Components**: Custom shadcn/ui implementation
- **Styling**: Tailwind CSS 3.4
- **Icons**: Lucide React
- **Animations**: Framer Motion
- **State Management**: TanStack React Query

## Quick Start

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

Runs the app at `http://localhost:3000`

### Build

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

## Pages

### Landing Page (`/`)

Product overview with:
- Animated hero section with floating particles
- System status badge
- Call-to-action buttons
- Tech stack indicators

### Agent Interface (`/agent`)

Primary interaction surface with:
- Document upload zone (drag & drop)
- File management sidebar
- Query input with submit
- Mode badges and system indicators

## Design System

### Colors

- **Background**: Deep navy to black gradient
- **Primary**: Cyan/Teal (#5EEADC)
- **Accent**: Emerald green
- **Text**: Off-white and slate tones

### Typography

- Font family: System sans-serif stack
- Weights: 400 (regular), 600 (semibold), 700 (bold)

### Components

All UI components follow shadcn/ui patterns with custom theming:

- `Button` - Multiple variants (default, outline, ghost)
- `Input` / `Textarea` - Form controls with focus states
- `Card` - Container component
- `Badge` - Status indicators

## Features

- ✅ Fully typed with TypeScript
- ✅ Responsive design
- ✅ Dark theme optimized
- ✅ Smooth animations
- ✅ File upload with drag & drop
- ✅ Zero placeholder/mock code
- ✅ Production-ready

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari 14+

## License

Part of the Nsure AI GraphRAG system.
