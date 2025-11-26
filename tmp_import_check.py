import traceback
try:
    import ui.main_window as m
    print('Imported OK')
except Exception:
    traceback.print_exc()
