/* Service Worker for Web Push */

self.addEventListener('install', function (event) {
    console.log('[Service Worker] Installing Service Worker ...', event);
    self.skipWaiting();
});

self.addEventListener('activate', function (event) {
    console.log('[Service Worker] Activating Service Worker ...', event);
    return self.clients.claim();
});

self.addEventListener('push', function (event) {
    console.log('[Service Worker] Push Received.');

    let payload = {};
    if (event.data) {
        try {
            payload = event.data.json();
        } catch (e) {
            payload = { title: '알림', body: event.data.text() };
        }
    }

    // 1. 현재 열려있는 창(클라이언트) 확인
    event.waitUntil(
        self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then(windowClients => {
            let clientIsFocused = false;

            for (let i = 0; i < windowClients.length; i++) {
                const client = windowClients[i];
                // 클라이언트에게 메시지 전송 (모달 띄우기용)
                client.postMessage({
                    type: 'PUSH_NOTIFICATION',
                    payload: payload
                });

                if (client.focused) {
                    clientIsFocused = true;
                }
            }

            // 2. 사용자가 보고 있으면(focused) 시스템 알림은 생략하고 모달만 띄움
            // (원하시면 아래 if문을 주석 처리해서 둘 다 뜨게 할 수도 있습니다)
            if (clientIsFocused) {
                console.log('[Service Worker] App is focused. Sending message instead of notification.');
                return;
            }

            // 3. 백그라운드 상태면 시스템 알림 표시
            const title = payload.title || 'Memorylary 알림';
            const options = {
                body: payload.body,
                vibrate: [100, 50, 100],
                data: {
                    dateOfArrival: Date.now(),
                    url: (payload.data && payload.data.url) ? payload.data.url : '/'
                },
                requireInteraction: true,
                actions: [
                    { action: 'explore', title: '확인하기' }
                ]
            };

            return self.registration.showNotification(title, options);
        })
    );
});

self.addEventListener('notificationclick', function (event) {
    console.log('[Service Worker] Notification click Received.');
    event.notification.close();

    // 클릭 시 URL 열기
    let targetUrl = '/';
    if (event.notification.data && event.notification.data.url) {
        targetUrl = event.notification.data.url;
    }

    event.waitUntil(
        clients.matchAll({ type: 'window' }).then(function (windowClients) {
            for (let i = 0; i < windowClients.length; i++) {
                const client = windowClients[i];
                if (client.url === targetUrl && 'focus' in client) {
                    return client.focus();
                }
            }
            if (clients.openWindow) {
                return clients.openWindow(targetUrl);
            }
        })
    );
});
