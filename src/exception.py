import sys


def error_message_detail(error_message, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f'File "{file_name}", line {exc_tb.tb_lineno}: {str(error_message)}.'
    return error_message


class SensorException(Exception):
    def __init__(self, error_message, error_detail: sys):
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
