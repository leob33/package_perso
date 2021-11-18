from logger_utils import create_custom_logger, log_function_call


def test_check_logger_return_something_in_terminal():
    logger = create_custom_logger('DEBUG')
    logger.info("I am the logger")


def test_check_log_function_call_print_something_in_terminal():
    logger = create_custom_logger('DEBUG')

    @log_function_call(logger)
    def sum(x, y):
        return x + y

    sum(12, y=1)


test_check_logger_return_something_in_terminal()
test_check_log_function_call_print_something_in_terminal()
