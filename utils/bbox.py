def calculate_iou(boxA, boxB):
    """
    Calcula la intersección sobre la unión (IoU) entre dos cajas delimitadoras.
    Cada caja se define como (x, y, w, h).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = boxAArea + boxBArea - interArea

    if unionArea == 0:
        return 0

    return interArea / unionArea


def remove_duplicate_billetes(billetes, iou_threshold=0.8):
    """
    Elimina billetes que se solapan demasiado (duplicados).
    """
    filtrados = []

    for billete in billetes:
        bbox = billete['bbox']
        duplicado = False

        for existente in filtrados:
            if calculate_iou(bbox, existente['bbox']) > iou_threshold:
                duplicado = True
                break

        if not duplicado:
            filtrados.append(billete)

    return filtrados
