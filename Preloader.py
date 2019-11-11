import Continuum
import sys
import time
from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import QApplication, QSplashScreen, QMainWindow

def main():
    app = QApplication(sys.argv)
    pixmap = QPixmap("assets/img/continuum_l2.png")
    splash = QSplashScreen(pixmap)
    splash.show()
    app.processEvents()
    window = QMainWindow()
    #window.show()
    #splash.finish( & window)
    time.sleep(10)
    splash.hide()
    Continuum.main()
    return app.exec_()

if __name__ == "__main__":
    main()