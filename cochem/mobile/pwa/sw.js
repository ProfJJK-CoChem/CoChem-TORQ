/**
 * CoChem Mobile PWA Service Worker (sw.js)
 * 
 * Invariants:
 * - Cache-First / Stale-While-Revalidate for immutable static assets (.js, .css, .woff2, .svg, .png).
 * - Strict Network-Only bypass for dynamic/telemetry/kernel endpoints (/api/kernels/*, /api/events/*, /api/telemetry/*, /api/push/*, ws://, wss://).
 * - Zero-Trust Push Notification presentation.
 */

const CACHE_NAME = 'cochem-mobile-shell-v1';

const STATIC_ASSET_EXTENSIONS = [
  '.js',
  '.css',
  '.woff2',
  '.woff',
  '.ttf',
  '.svg',
  '.png',
  '.jpg',
  '.jpeg',
  '.ico',
  '.webmanifest',
  '.json'
];

const NETWORK_ONLY_PATTERNS = [
  /\/api\/kernels\//i,
  /\/api\/events\//i,
  /\/api\/telemetry\//i,
  /\/api\/push\//i,
  /\/api\/sessions\//i,
  /\/api\/status\//i,
  /\/ws\//i,
  /\/socket\.io\//i
];

// Pre-cached offline core shell assets
const PRECACHE_URLS = [
  '/',
  '/manifest.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(PRECACHE_URLS).catch((err) => {
        console.warn('[CoChem SW] Pre-cache non-fatal warning:', err);
      });
    }).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      );
    }).then(() => self.clients.claim())
  );
});

/**
 * Determine if a request URL matches static asset caching criteria.
 */
function isStaticAsset(url) {
  const pathname = url.pathname.toLowerCase();
  return STATIC_ASSET_EXTENSIONS.some((ext) => pathname.endsWith(ext));
}

/**
 * Determine if a request URL requires strict Network-Only bypass.
 */
function isNetworkOnly(url) {
  if (url.protocol === 'ws:' || url.protocol === 'wss:') {
    return true;
  }
  const fullPath = url.pathname + url.search;
  return NETWORK_ONLY_PATTERNS.some((pattern) => pattern.test(fullPath));
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  const url = new URL(request.url);

  // 1. WebSocket or Non-HTTP(S) requests - pass through
  if (request.method !== 'GET') {
    event.respondWith(fetch(request));
    return;
  }

  // 2. Strict Network-Only bypass for dynamic/telemetry/kernel routes
  if (isNetworkOnly(url)) {
    event.respondWith(
      fetch(request, { cache: 'no-store' }).catch((err) => {
        return new Response(
          JSON.stringify({ error: 'Network-only route offline', details: err.message }),
          { status: 503, headers: { 'Content-Type': 'application/json' } }
        );
      })
    );
    return;
  }

  // 3. Cache-First / Stale-While-Revalidate for static assets
  if (isStaticAsset(url)) {
    event.respondWith(
      caches.open(CACHE_NAME).then((cache) => {
        return cache.match(request).then((cachedResponse) => {
          const fetchPromise = fetch(request).then((networkResponse) => {
            if (networkResponse && networkResponse.status === 200) {
              cache.put(request, networkResponse.clone());
            }
            return networkResponse;
          }).catch(() => cachedResponse);

          return cachedResponse || fetchPromise;
        });
      })
    );
    return;
  }

  // 4. Default: Network-First with Cache Fallback for navigation requests
  event.respondWith(
    fetch(request)
      .then((networkResponse) => {
        if (networkResponse && networkResponse.status === 200) {
          const responseClone = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(request, responseClone));
        }
        return networkResponse;
      })
      .catch(() => {
        return caches.match(request).then((cachedResponse) => {
          return cachedResponse || caches.match('/');
        });
      })
  );
});

// Push notification event listener
self.addEventListener('push', (event) => {
  let data = {
    title: 'CoChem Alert',
    body: 'Calculation event received.',
    icon: '/static/icons/cochem-icon-192.png',
    tag: 'cochem-push'
  };

  if (event.data) {
    try {
      const json = event.data.json();
      data.title = json.title || `CoChem: Job ${json.status || 'Update'}`;
      data.body = json.summary || `Exit code: ${json.exit_code ?? 0}`;
      if (json.job_id) {
        data.tag = `cochem-job-${json.job_id}`;
      }
    } catch (e) {
      data.body = event.data.text();
    }
  }

  const options = {
    body: data.body,
    icon: data.icon,
    badge: '/static/icons/cochem-icon-192.png',
    tag: data.tag,
    renotify: true,
    vibrate: [200, 100, 200]
  };

  event.waitUntil(
    self.registration.showNotification(data.title, options)
  );
});

// Push notification click handler
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clientList) => {
      for (const client of clientList) {
        if (client.url && 'focus' in client) {
          return client.focus();
        }
      }
      if (clients.openWindow) {
        return clients.openWindow('/');
      }
    })
  );
});
