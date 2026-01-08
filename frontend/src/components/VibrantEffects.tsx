import { motion } from "framer-motion";
import { useEffect, useState } from "react";

interface OrbitingCircle {
  id: number;
  radius: number;
  duration: number;
  delay: number;
  color: string;
}

export function OrbitingElements() {
  const [circles] = useState<OrbitingCircle[]>([
    { id: 1, radius: 150, duration: 20, delay: 0, color: "rgba(94, 234, 212, 0.3)" },
    { id: 2, radius: 200, duration: 25, delay: 2, color: "rgba(139, 92, 246, 0.3)" },
    { id: 3, radius: 250, duration: 30, delay: 4, color: "rgba(236, 72, 153, 0.3)" },
  ]);

  return (
    <div className="absolute inset-0 flex items-center justify-center pointer-events-none overflow-hidden">
      {circles.map((circle) => (
        <motion.div
          key={circle.id}
          className="absolute"
          animate={{
            rotate: 360,
          }}
          transition={{
            duration: circle.duration,
            delay: circle.delay,
            repeat: Infinity,
            ease: "linear",
          }}
        >
          <div
            className="rounded-full blur-xl"
            style={{
              width: `${circle.radius}px`,
              height: `${circle.radius}px`,
              border: `2px solid ${circle.color}`,
              boxShadow: `0 0 30px ${circle.color}`,
            }}
          />
        </motion.div>
      ))}
    </div>
  );
}

interface WaveProps {
  delay?: number;
  color?: string;
}

export function AnimatedWave({ delay = 0, color = "rgba(94, 234, 212, 0.1)" }: WaveProps) {
  return (
    <motion.div
      className="absolute bottom-0 left-0 right-0 h-32"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay }}
    >
      <svg
        className="absolute bottom-0 w-full h-full"
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 1440 320"
        preserveAspectRatio="none"
      >
        <motion.path
          fill={color}
          d="M0,160L48,144C96,128,192,96,288,106.7C384,117,480,171,576,186.7C672,203,768,181,864,154.7C960,128,1056,96,1152,101.3C1248,107,1344,149,1392,170.7L1440,192L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"
          animate={{
            d: [
              "M0,160L48,144C96,128,192,96,288,106.7C384,117,480,171,576,186.7C672,203,768,181,864,154.7C960,128,1056,96,1152,101.3C1248,107,1344,149,1392,170.7L1440,192L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z",
              "M0,192L48,197.3C96,203,192,213,288,197.3C384,181,480,139,576,128C672,117,768,139,864,154.7C960,171,1056,181,1152,170.7C1248,160,1344,128,1392,112L1440,96L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z",
              "M0,160L48,144C96,128,192,96,288,106.7C384,117,480,171,576,186.7C672,203,768,181,864,154.7C960,128,1056,96,1152,101.3C1248,107,1344,149,1392,170.7L1440,192L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z",
            ],
          }}
          transition={{
            duration: 10,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      </svg>
    </motion.div>
  );
}

export function FloatingShapes() {
  const shapes = [
    { id: 1, size: 60, x: 10, y: 20, duration: 8, color: "from-cyan-400/20 to-teal-400/20" },
    { id: 2, size: 80, x: 80, y: 30, duration: 10, color: "from-violet-400/20 to-purple-400/20" },
    { id: 3, size: 50, x: 70, y: 70, duration: 12, color: "from-pink-400/20 to-rose-400/20" },
    { id: 4, size: 70, x: 20, y: 80, duration: 9, color: "from-emerald-400/20 to-green-400/20" },
    { id: 5, size: 55, x: 50, y: 50, duration: 11, color: "from-blue-400/20 to-cyan-400/20" },
  ];

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {shapes.map((shape) => (
        <motion.div
          key={shape.id}
          className={`absolute bg-gradient-to-br ${shape.color} rounded-full blur-2xl`}
          style={{
            width: `${shape.size}px`,
            height: `${shape.size}px`,
            left: `${shape.x}%`,
            top: `${shape.y}%`,
          }}
          animate={{
            x: [0, 30, -30, 0],
            y: [0, -30, 30, 0],
            scale: [1, 1.2, 0.8, 1],
            rotate: [0, 90, 180, 270, 360],
          }}
          transition={{
            duration: shape.duration,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
}
