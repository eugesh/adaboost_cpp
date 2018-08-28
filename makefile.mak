base_dir=$(BASE_DIR)
include $(MAKE_FOR_SYSTEM)

##############################################

# имя исполняемого файла
name_exe=abc_class_lib.dll

# директория с исходными файлами
source_dir=$(base_dir)/source

# директория для объектных файлов
obj_dir=$(base_dir)/obj

# директория для moc-файлов
moc_dir=$(base_dir)/moc

# директория для выполняемого файла
proc_dir=$(base_dir)/exe

# директория с h-файлами оконных форм
source_header_form_dir=$(base_dir)/form_h

# директория с ui-файлами оконных форм
source_form_dir=$(base_dir)/form_ui

# директория с заголовочными файлами
cpp_options_include=$(opt_inc) "$(source_dir)" \
							$(opt_inc) "$(source_header_form_dir)"

# h-файлы с формами для окон
#h_file_widget=\
#	$(source_header_form_dir)/ui_viewer.h

# объектные файла за исключением moc-файлов
obj_file=\
	$(obj_dir)/adaboost_cumsum_lib.$(ext_obj)
#	$(obj_dir)/main.$(ext_obj) \

# объектные файлы полученные из moc-файлов						
#obj_moc_file=\
#	$(obj_dir)/cviewer_moc.$(ext_obj) \
#	$(obj_dir)/cscene_moc.$(ext_obj)

all:
	$(MAKE) $(opt_make_f) makefile.mak clean
	$(MAKE) $(opt_make_f) makefile.mak program

clean:
	$(command_del) $(base_dir)$(SD)obj$(SD)*.$(ext_obj)
	$(command_del) $(base_dir)$(SD)moc$(SD)*moc.cpp
	$(command_del) $(obj_file)
	$(command_del) $(obj_moc_file)
	$(command_del) $(proc_dir)/$(name_exe)

# описание цели "построение приложения"
program :  $(proc_dir)/$(name_exe)

# создание h-файлов на основе ui-файлов
#$(source_header_form_dir)/ui_viewer.h : $(source_form_dir)/ui_viewer.ui
#	$(UIC) -o $@ $?

# создание moc-файлов
#$(moc_dir)/cviewer_moc.cpp : $(source_dir)/cviewer.h
#	$(MOC) -o $@ $?

#$(moc_dir)/cscene_moc.cpp : $(source_dir)/cscene.h
#	$(MOC) -o $@ $? 

# исходные файлы

#$(obj_dir)/main.$(ext_obj) : $(source_dir)/main.cpp
#	$(command_comp)

$(obj_dir)/adaboost_cumsum_lib.$(ext_obj) : $(source_dir)/adaboost_cumsum_lib.cpp
	$(command_comp)

$(source_dir)/adaboost_cumsum_lib.cpp : \
		$(source_dir)/adaboost_cumsum_lib.h
	$(command_touch)

######################################################

#  подключение дополнительных моделей
#include $(base_dir)/makefile_lib.mak

######################################################

# создание приложения с помощью линковщика
$(proc_dir)/$(name_exe) : $(obj_file)
	$(command_linkDLL)  $(obj_file) $(library)


