
target("ppl")
    set_kind("static")
    add_includedirs("$(projectdir)/include")
    add_headerfiles("../include/**/*")
    add_files("host/*.cpp")
    add_packages("glm", "pthread")

target("ppl-hybrid")
    set_kind("static")
    add_includedirs("$(projectdir)/include")
    add_files(
        "cuda/*.cu",
        "host/02_sort.cpp",
        "host/dispatcher.cpp"
        -- Must exclude "host/structures.cpp"
        )
    add_packages("glm")
    add_cugencodes("native")
