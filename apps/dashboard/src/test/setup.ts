import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

// Each test starts from an empty document, real timers and the real fetch.
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.useRealTimers(); sessionStorage.clear(); });
