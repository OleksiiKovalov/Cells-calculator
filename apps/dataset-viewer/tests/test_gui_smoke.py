"""Off-screen GUI smoke test: open a dataset, select an image, toggle overlay."""
from main_window import MainWindow
from tests.conftest import sample


def test_open_select_and_toggle(qapp):
    win = MainWindow()
    win._open_path(sample("yolo_seg"))
    assert win._loader is not None

    items = win.browser._selectable_items()
    assert len(items) == 3  # 3 images across train/val

    win.browser.tree.setCurrentItem(items[0])
    qapp.processEvents()
    assert win.viewer._pixmap_item is not None      # image rendered
    assert win.viewer._ann_items                     # annotations drawn

    # Menu/toolbar annotation toggles stay in sync (the _set_annotations fix).
    win._set_annotations(False)
    assert not win._ann_action.isChecked()
    assert not win._tb_ann_action.isChecked()
    win._set_annotations(True)
    assert win._ann_action.isChecked()
    assert win._tb_ann_action.isChecked()
