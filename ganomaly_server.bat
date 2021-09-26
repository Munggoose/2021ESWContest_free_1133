@echo on
set root=C:\anaconda3
call %root%\Scripts\activate.bat %root%

call activate mun

:START

call python server.py

@GOTO START