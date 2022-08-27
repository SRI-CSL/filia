message("Hello $ENV{SOURCE}")
set(CONFIG_PY_DIR $ENV{PY_DIR})
configure_file($ENV{SOURCE} "$ENV{TARGET}" @ONLY)
