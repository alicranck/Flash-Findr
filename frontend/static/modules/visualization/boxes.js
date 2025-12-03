// Boxes Visualization Module - Handles bounding box drawing

export function drawBoxes(layer, data, scaleX, scaleY, offsetX, offsetY, pointer) {
    if (!data.boxes) return;

    const colors = ['#00f2ff', '#7000ff', '#00ff9d', '#ffbd2e', '#ff0055'];

    function getColor(idx) {
        return colors[idx % colors.length];
    }

    data.boxes.forEach(box => {
        const [x1, y1, x2, y2] = box.xyxy;
        const w = (x2 - x1) * scaleX;
        const h = (y2 - y1) * scaleY;
        const x = x1 * scaleX + offsetX;
        const y = y1 * scaleY + offsetY;

        const color = getColor(box.cls);

        // Check for hover
        let isHover = false;
        if (pointer) {
            if (pointer.x >= x && pointer.x <= x + w &&
                pointer.y >= y && pointer.y <= y + h) {
                isHover = true;
            }
        }

        // Rect
        const rect = new Konva.Rect({
            x: x,
            y: y,
            width: w,
            height: h,
            stroke: color,
            strokeWidth: isHover ? 4 : 2,
            shadowColor: color,
            shadowBlur: isHover ? 15 : 0,
            name: 'bbox'
        });

        if (isHover) {
            const tooltip = new Konva.Label({
                x: x,
                y: y - 10,
                opacity: 1
            });

            tooltip.add(new Konva.Tag({
                fill: color,
                pointerDirection: 'down',
                pointerWidth: 10,
                pointerHeight: 10,
                lineJoin: 'round',
                shadowColor: 'black',
                shadowBlur: 10,
                shadowOffset: { x: 10, y: 10 },
                shadowOpacity: 0.5
            }));

            const className = (data.class_names && data.class_names[box.cls]) ? data.class_names[box.cls] : box.cls;
            tooltip.add(new Konva.Text({
                text: `${className} ${box.id}: ${box.conf.toFixed(2)}`,
                fontFamily: 'monospace',
                fontSize: 14,
                padding: 5,
                fill: 'black'
            }));

            layer.add(rect);
            layer.add(tooltip);
        } else {
            layer.add(rect);
        }
    });
}
