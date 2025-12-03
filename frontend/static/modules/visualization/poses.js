// Poses Visualization Module - Handles pose skeleton drawing

export function drawPoses(layer, data, scaleX, scaleY, offsetX, offsetY) {
    if (!data.poses) return;

    // Define skeleton connections with colors for different body parts
    const skeletonConnections = [
        // Head (cyan)
        { pair: [0, 1], color: '#00ffff' },
        { pair: [0, 2], color: '#00ffff' },
        { pair: [1, 3], color: '#00ffff' },
        { pair: [2, 4], color: '#00ffff' },

        // Torso (yellow)
        { pair: [5, 6], color: '#ffff00' },
        { pair: [5, 11], color: '#ffff00' },
        { pair: [6, 12], color: '#ffff00' },
        { pair: [11, 12], color: '#ffff00' },

        // Left arm (magenta)
        { pair: [5, 7], color: '#ff00ff' },
        { pair: [7, 9], color: '#ff00ff' },

        // Right arm (orange)
        { pair: [6, 8], color: '#ff8800' },
        { pair: [8, 10], color: '#ff8800' },

        // Left leg (green)
        { pair: [11, 13], color: '#00ff00' },
        { pair: [13, 15], color: '#00ff00' },

        // Right leg (lime)
        { pair: [12, 14], color: '#88ff00' },
        { pair: [14, 16], color: '#88ff00' }
    ];

    // Keypoint colors by index
    const keypointColors = [
        '#00ffff', '#00ffff', '#00ffff', '#00ffff', '#00ffff', // Head (0-4)
        '#ff00ff', '#ff8800', // Shoulders (5-6)
        '#ff00ff', '#ff8800', // Elbows (7-8)
        '#ff00ff', '#ff8800', // Wrists (9-10)
        '#00ff00', '#88ff00', // Hips (11-12)
        '#00ff00', '#88ff00', // Knees (13-14)
        '#00ff00', '#88ff00'  // Ankles (15-16)
    ];

    data.poses.forEach(pose => {
        const kpts = pose.keypoints; // List of [[x,y], conf]

        // Draw Skeleton
        skeletonConnections.forEach(conn => {
            const idx1 = conn.pair[0];
            const idx2 = conn.pair[1];

            if (idx1 < kpts.length && idx2 < kpts.length) {
                const kp1 = kpts[idx1];
                const kp2 = kpts[idx2];

                if (kp1[1] > 0.5 && kp2[1] > 0.5) { // Check confidence
                    const line = new Konva.Line({
                        points: [
                            kp1[0][0] * scaleX + offsetX, kp1[0][1] * scaleY + offsetY,
                            kp2[0][0] * scaleX + offsetX, kp2[0][1] * scaleY + offsetY
                        ],
                        stroke: conn.color,
                        strokeWidth: 3,
                        lineCap: 'round',
                        lineJoin: 'round',
                        opacity: 0.8
                    });
                    layer.add(line);
                }
            }
        });

        // Draw Keypoints
        kpts.forEach((kp, idx) => {
            const [x, y] = kp[0];
            const conf = kp[1];
            if (conf > 0.5) {
                const circle = new Konva.Circle({
                    x: x * scaleX + offsetX,
                    y: y * scaleY + offsetY,
                    radius: 4,
                    fill: keypointColors[idx] || '#ffffff',
                    stroke: 'white',
                    strokeWidth: 1.5,
                    shadowColor: 'black',
                    shadowBlur: 3,
                    shadowOpacity: 0.5
                });
                layer.add(circle);
            }
        });
    });
}
