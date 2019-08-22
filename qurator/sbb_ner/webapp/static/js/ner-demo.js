$(document).ready(function(){

    $('#nerform').submit(
        function(e){
            e.preventDefault();

            var task = $('#task').val();
            var model_id = $('#model').val();
            var input_text = $('#inputtext').val()

            do_task(task, model_id, input_text);
        }
    );

    $.get( "/models")
        .done(
            function( data ) {
                var tmp="";
                $.each(data,
                    function(index, item){

                        selected=""
                        if (item.default) {
                            selected = "selected"
                        }

                        tmp += '<option value="' + item.id + '" ' + selected + ' >' + item.name + '</option>'
                    });
                    $('#model').html(tmp);
                }
            );

    task_select()
});
