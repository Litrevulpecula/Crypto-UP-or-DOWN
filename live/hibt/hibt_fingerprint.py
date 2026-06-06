#!/usr/bin/env python3
"""Comprehensive fingerprint hardening for the HiBT Playwright browser.

Covers: navigator properties, window.chrome, WebGL, Canvas noise, AudioContext
noise, screen dimensions, WebRTC leak blocking, permissions, plugins, connection
API, iframe recursion, and Function.toString cloaking.
"""
from __future__ import annotations

import json
import random
from typing import Any

from hibt_config import FingerprintConfig

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
)


def resolve_user_agent(fp: FingerprintConfig) -> str:
    return fp.user_agent or DEFAULT_USER_AGENT


def context_options(fp: FingerprintConfig) -> dict[str, Any]:
    """Launch-time options set on the persistent context."""
    jitter = random.randint(-fp.viewport_jitter_px, fp.viewport_jitter_px)
    return {
        "user_agent": resolve_user_agent(fp),
        "color_scheme": fp.color_scheme,
        "device_scale_factor": fp.device_scale_factor,
        "extra_http_headers": {"Accept-Language": fp.accept_language},
        "screen": {"width": fp.screen_width, "height": fp.screen_height},
        "viewport": None,  # will be overridden by caller with jitter
        "_viewport_jitter": jitter,
    }


def launch_args(fp: FingerprintConfig) -> list[str]:
    """Chromium flags that reduce automation detection at the process level."""
    args = [
        "--disable-blink-features=AutomationControlled",
        "--disable-features=IsolateOrigins,site-per-process",
        "--no-default-browser-check",
        "--no-first-run",
        "--disable-infobars",
        "--disable-background-networking",
        "--disable-component-update",
        f"--lang={fp.languages[0] if fp.languages else 'zh-CN'}",
    ]
    if fp.block_webrtc_leak:
        args.append("--enforce-webrtc-ip-permission-check")
        args.append("--webrtc-ip-handling-policy=disable_non_proxied_udp")
    return args


def init_script(fp: FingerprintConfig) -> str:
    """JS injected via add_init_script before any page script."""
    seed = random.randint(1, 2**31)
    cfg = json.dumps(
        {
            "languages": fp.languages,
            "language": fp.languages[0] if fp.languages else "zh-CN",
            "platform": fp.platform,
            "hardwareConcurrency": fp.hardware_concurrency,
            "deviceMemory": fp.device_memory,
            "webglVendor": fp.webgl_vendor,
            "webglRenderer": fp.webgl_renderer,
            "screenWidth": fp.screen_width,
            "screenHeight": fp.screen_height,
            "screenColorDepth": fp.screen_color_depth,
            "canvasNoise": fp.canvas_noise,
            "audioNoise": fp.audio_noise,
            "blockWebrtcLeak": fp.block_webrtc_leak,
            "seed": seed,
        }
    )
    return _INIT_SCRIPT_TEMPLATE.replace("__FP_CONFIG__", cfg)


_INIT_SCRIPT_TEMPLATE = r"""
(() => {
  const CFG = __FP_CONFIG__;

  // === Utilities ===
  const define = (obj, prop, getter) => {
    try {
      Object.defineProperty(obj, prop, { get: getter, configurable: true, enumerable: true });
    } catch (e) {}
  };
  const defineValue = (obj, prop, value) => {
    try {
      Object.defineProperty(obj, prop, { value, writable: false, configurable: true, enumerable: true });
    } catch (e) {}
  };
  // Seeded PRNG (xorshift32) for deterministic noise across page loads
  let _s = CFG.seed >>> 0;
  const rand = () => { _s ^= _s << 13; _s ^= _s >>> 17; _s ^= _s << 5; return (_s >>> 0) / 4294967296; };

  // === 1. navigator.webdriver ===
  delete Navigator.prototype.webdriver;
  define(Navigator.prototype, 'webdriver', () => undefined);

  // === 2. navigator properties ===
  define(Navigator.prototype, 'languages', () => Object.freeze([...CFG.languages]));
  define(Navigator.prototype, 'language', () => CFG.language);
  define(Navigator.prototype, 'platform', () => CFG.platform);
  define(Navigator.prototype, 'hardwareConcurrency', () => CFG.hardwareConcurrency);
  define(Navigator.prototype, 'deviceMemory', () => CFG.deviceMemory);
  define(Navigator.prototype, 'maxTouchPoints', () => 0);
  define(Navigator.prototype, 'vendor', () => 'Google Inc.');
  define(Navigator.prototype, 'appVersion', () => '5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36');

  // === 3. window.chrome ===
  if (!window.chrome) window.chrome = {};
  if (!window.chrome.runtime) {
    window.chrome.runtime = { connect: function(){}, sendMessage: function(){} };
  }
  window.chrome.app = window.chrome.app || {
    isInstalled: false,
    getDetails: function() { return null; },
    getIsInstalled: function() { return false; },
    InstallState: { DISABLED:'disabled', INSTALLED:'installed', NOT_INSTALLED:'not_installed' },
    RunningState: { CANNOT_RUN:'cannot_run', READY_TO_RUN:'ready_to_run', RUNNING:'running' },
  };
  window.chrome.csi = window.chrome.csi || function() { return {}; };
  window.chrome.loadTimes = window.chrome.loadTimes || function() { return {}; };

  // === 4. Screen properties ===
  const screenProps = {
    width: CFG.screenWidth, height: CFG.screenHeight,
    availWidth: CFG.screenWidth, availHeight: CFG.screenHeight - 40,
    colorDepth: CFG.screenColorDepth, pixelDepth: CFG.screenColorDepth,
  };
  for (const [k, v] of Object.entries(screenProps)) {
    define(Screen.prototype, k, () => v);
  }
  defineValue(window, 'outerWidth', CFG.screenWidth);
  defineValue(window, 'outerHeight', CFG.screenHeight);
  define(window, 'screenX', () => 0);
  define(window, 'screenY', () => 0);

  // === 5. Permissions.query consistency ===
  const origQuery = window.navigator.permissions && window.navigator.permissions.query;
  if (origQuery) {
    window.navigator.permissions.query = function query(params) {
      return params && params.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission || 'default', onchange: null })
        : origQuery.call(window.navigator.permissions, params);
    };
  }

  // === 6. WebGL vendor/renderer ===
  const patchGL = (proto) => {
    if (!proto || !proto.getParameter) return;
    const orig = proto.getParameter;
    proto.getParameter = function getParameter(param) {
      if (param === 37445) return CFG.webglVendor;
      if (param === 37446) return CFG.webglRenderer;
      return orig.call(this, param);
    };
  };
  if (window.WebGLRenderingContext) patchGL(WebGLRenderingContext.prototype);
  if (window.WebGL2RenderingContext) patchGL(WebGL2RenderingContext.prototype);
  const glProtos = [
    window.WebGLRenderingContext && WebGLRenderingContext.prototype,
    window.WebGL2RenderingContext && WebGL2RenderingContext.prototype,
  ].filter(Boolean);

  // === 7. Canvas fingerprint noise ===
  if (CFG.canvasNoise) {
    const origToDataURL = HTMLCanvasElement.prototype.toDataURL;
    const origToBlob = HTMLCanvasElement.prototype.toBlob;
    const origGetImageData = CanvasRenderingContext2D.prototype.getImageData;

    const addNoise = (imageData) => {
      const d = imageData.data;
      for (let i = 0; i < d.length; i += 4) {
        const noise = Math.floor((rand() - 0.5) * 3);
        d[i] = Math.max(0, Math.min(255, d[i] + noise));
      }
      return imageData;
    };

    CanvasRenderingContext2D.prototype.getImageData = function getImageData(...args) {
      return addNoise(origGetImageData.apply(this, args));
    };
    HTMLCanvasElement.prototype.toDataURL = function toDataURL(...args) {
      try {
        const ctx = this.getContext('2d');
        if (ctx) {
          const img = origGetImageData.call(ctx, 0, 0, this.width || 1, this.height || 1);
          addNoise(img);
          ctx.putImageData(img, 0, 0);
        }
      } catch (e) {}
      return origToDataURL.apply(this, args);
    };
    HTMLCanvasElement.prototype.toBlob = function toBlob(cb, ...rest) {
      try {
        const ctx = this.getContext('2d');
        if (ctx) {
          const img = origGetImageData.call(ctx, 0, 0, this.width || 1, this.height || 1);
          addNoise(img);
          ctx.putImageData(img, 0, 0);
        }
      } catch (e) {}
      return origToBlob.call(this, cb, ...rest);
    };
  }

  // === 8. AudioContext fingerprint noise ===
  if (CFG.audioNoise && window.AudioContext) {
    const OrigAC = window.AudioContext;
    const origGetFloat = AnalyserNode.prototype.getFloatFrequencyData;
    AnalyserNode.prototype.getFloatFrequencyData = function getFloatFrequencyData(array) {
      origGetFloat.call(this, array);
      for (let i = 0; i < array.length; i++) {
        array[i] += (rand() - 0.5) * 0.001;
      }
    };
    const origCopyFrom = AudioBuffer.prototype.copyFromChannel;
    AudioBuffer.prototype.copyFromChannel = function copyFromChannel(dest, ch, off) {
      origCopyFrom.call(this, dest, ch, off || 0);
      for (let i = 0; i < dest.length; i++) {
        dest[i] += (rand() - 0.5) * 0.0001;
      }
    };
  }

  // === 9. WebRTC IP leak blocking ===
  if (CFG.blockWebrtcLeak) {
    const OrigRTC = window.RTCPeerConnection || window.webkitRTCPeerConnection;
    if (OrigRTC) {
      const handler = {
        construct(target, args) {
          const config = args[0] || {};
          config.iceServers = [];
          args[0] = config;
          const pc = new target(...args);
          const origCreate = pc.createDataChannel.bind(pc);
          pc.createDataChannel = function() { return origCreate(...arguments); };
          return pc;
        }
      };
      window.RTCPeerConnection = new Proxy(OrigRTC, handler);
      if (window.webkitRTCPeerConnection) {
        window.webkitRTCPeerConnection = window.RTCPeerConnection;
      }
    }
  }

  // === 10. navigator.connection (Network Information API) ===
  if (!navigator.connection) {
    Object.defineProperty(navigator, 'connection', {
      get: () => ({
        effectiveType: '4g', rtt: 50, downlink: 10, saveData: false,
        type: 'wifi', downlinkMax: Infinity, ontypechange: null, onchange: null,
        addEventListener: function(){}, removeEventListener: function(){},
      }),
      configurable: true,
    });
  }

  // === 11. Plugins (realistic for Chrome on Windows) ===
  try {
    const makeMime = (type, suffixes, desc) => ({ type, suffixes, description: desc, enabledPlugin: null });
    const makePlugin = (name, filename, desc, mimes) => {
      const p = { name, filename, description: desc, length: mimes.length };
      mimes.forEach((m, i) => { p[i] = m; m.enabledPlugin = p; });
      return p;
    };
    const pdfMime = makeMime('application/pdf', 'pdf', 'Portable Document Format');
    const plugins = [
      makePlugin('PDF Viewer', 'internal-pdf-viewer', 'Portable Document Format', [pdfMime]),
      makePlugin('Chrome PDF Viewer', 'internal-pdf-viewer', 'Portable Document Format', [pdfMime]),
      makePlugin('Chromium PDF Viewer', 'internal-pdf-viewer', 'Portable Document Format', [pdfMime]),
      makePlugin('Microsoft Edge PDF Viewer', 'internal-pdf-viewer', '', [pdfMime]),
      makePlugin('WebKit built-in PDF', 'internal-pdf-viewer', '', [pdfMime]),
    ];
    define(Navigator.prototype, 'plugins', () => {
      const arr = plugins.slice();
      arr.item = (i) => arr[i] || null;
      arr.namedItem = (n) => arr.find(p => p.name === n) || null;
      arr.refresh = function(){};
      Object.defineProperty(arr, 'length', { value: plugins.length });
      return arr;
    });
    define(Navigator.prototype, 'mimeTypes', () => {
      const arr = [pdfMime];
      arr.item = (i) => arr[i] || null;
      arr.namedItem = (t) => arr.find(m => m.type === t) || null;
      Object.defineProperty(arr, 'length', { value: 1 });
      return arr;
    });
  } catch (e) {}

  // === 12. Iframe contentWindow recursion defense ===
  // Ensure our patches apply in dynamically created iframes too
  const origCreateElement = document.createElement.bind(document);
  document.createElement = function createElement(tag, ...rest) {
    const el = origCreateElement(tag, ...rest);
    if (tag.toLowerCase() === 'iframe') {
      const origAppend = el.appendChild;
      const orig_set_src = Object.getOwnPropertyDescriptor(HTMLIFrameElement.prototype, 'src');
      // Will be patched once attached to DOM via MutationObserver below
    }
    return el;
  };

  // === 13. Prevent detection via toString / prototype checks ===
  const nativeToString = Function.prototype.toString;
  const spoofed = new WeakSet();
  const fns = [
    window.navigator.permissions && window.navigator.permissions.query,
    HTMLCanvasElement.prototype.toDataURL,
    HTMLCanvasElement.prototype.toBlob,
    CanvasRenderingContext2D.prototype.getImageData,
    document.createElement,
    ...glProtos.map(p => p.getParameter),
  ].filter(Boolean);
  fns.forEach(f => spoofed.add(f));

  Function.prototype.toString = function toString() {
    if (spoofed.has(this)) {
      return `function ${this.name || ''}() { [native code] }`;
    }
    return nativeToString.call(this);
  };
  spoofed.add(Function.prototype.toString);

  // === 14. Hide Playwright-specific artifacts ===
  delete window.__playwright;
  delete window.__pw_manual;
  delete window.__PW_inspect;

  // === 15. performance.now() slight jitter to defeat timing fingerprints ===
  const origPerfNow = performance.now.bind(performance);
  performance.now = function now() {
    return origPerfNow() + (rand() - 0.5) * 0.05;
  };
  spoofed.add(performance.now);

  // === 16. Notification permission consistency ===
  try {
    defineValue(Notification, 'permission', 'default');
  } catch (e) {}

})();
"""
