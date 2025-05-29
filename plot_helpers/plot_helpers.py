from matplotlib import get_backend

def move_figure(f, x, y):
    pass
    # """Move figure's upper left corner to pixel (x, y)"""
    # backend = get_backend()
    # print("Backend", backend)
    # if backend.lower() == 'tkagg':
    #     f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    # elif backend.lower() == 'wxagg':
    #     f.canvas.manager.window.SetPosition((x, y))
    # else:
    #     # This works for QT and GTK
    #     # You can also use window.setGeometry
    #     f.canvas.manager.window.move(x, y)