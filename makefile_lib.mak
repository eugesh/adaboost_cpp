bj_file+=\
			$(obj_dir)/harrisdetector.$(ext_obj) \
			$(obj_dir)/auxqimage.$(ext_obj) \
			$(obj_dir)/cdevbfsearch.$(ext_obj)


##########################################################################

$(obj_dir)/harrisdetector.$(ext_obj) : $(source_mylib_matching_dir)/harrisdetector.cpp
	$(command_comp)

$(obj_dir)/auxqimage.$(ext_obj) : $(source_mylib_aux_dir)/auxqimage.cpp
	$(command_comp)

#$(obj_dir)/cdevbfsearch.$(ext_obj) : $(source_dir)/cdevbfsearch.cpp
#	$(command_comp)


##########################################################################

