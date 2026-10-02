"use client";

import {
  useEffect,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from "react";
import { CrtBackground } from "../shaders/crt/CrtBackground";

const INTRO_DURATION_MS = 3200;
const FADE_DURATION_MS = 650;
const REDUCED_MOTION_QUERY = "(prefers-reduced-motion: reduce)";

function subscribeMotionPreference(onChange: () => void) {
  const preference = window.matchMedia(REDUCED_MOTION_QUERY);
  preference.addEventListener("change", onChange);
  return () => preference.removeEventListener("change", onChange);
}

function getMotionPreference() {
  return window.matchMedia(REDUCED_MOTION_QUERY).matches;
}

export function CrtIntro({ children }: { children: ReactNode }) {
  const [isVisible, setIsVisible] = useState(true);
  const [isLeaving, setIsLeaving] = useState(false);
  const prefersReducedMotion = useSyncExternalStore(
    subscribeMotionPreference,
    getMotionPreference,
    () => false,
  );
  const introActive = isVisible && !prefersReducedMotion;

  function revealPage() {
    if (isLeaving) return;
    setIsLeaving(true);
  }

  useEffect(() => {
    if (prefersReducedMotion) return undefined;

    const timer = window.setTimeout(() => setIsLeaving(true), INTRO_DURATION_MS);
    return () => window.clearTimeout(timer);
  }, [prefersReducedMotion]);

  useEffect(() => {
    if (!isLeaving) return undefined;

    const timer = window.setTimeout(
      () => setIsVisible(false),
      FADE_DURATION_MS,
    );
    return () => window.clearTimeout(timer);
  }, [isLeaving]);

  return (
    <>
      <div
        className={`intro-page${!introActive || isLeaving ? " intro-page--visible" : ""}`}
        aria-hidden={introActive && !isLeaving}
        inert={introActive && !isLeaving}
      >
        {children}
      </div>

      {introActive && (
        <div
          className={`crt-intro-layer${isLeaving ? " crt-intro-layer--fading" : ""}`}
        >
          <CrtBackground
            className="crt-intro-background"
            variant="terminal"
            speed={1.0}
            typeSpeed={1.0}
            motion={1.0}
            hue={0}
            saturation={1.0}
            brightness={1.0}
            opacity={1.0}
          />
          <button
            className="crt-intro-skip"
            type="button"
            onClick={revealPage}
            disabled={isLeaving}
          >
            Skip intro
          </button>
        </div>
      )}
    </>
  );
}