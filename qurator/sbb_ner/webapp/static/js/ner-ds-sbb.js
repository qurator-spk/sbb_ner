
$(document).ready(function(){

    $('#nerform').submit(
        function(e){
            e.preventDefault();

            var task = $('#task').val();
            var model_id = $('#model').val();

            var spinner_html =
                `<div class="d-flex justify-content-center">
                    <div class="spinner-border align-center" role="status">
                        <span class="sr-only">Loading...</span>
                    </div>
                 </div>`;

            var ppn = $('#ppn').val()

            $("#resultregion").html(spinner_html);

            $.get( "/digisam-fulltext/" + ppn)
                .done(function( data ) {

                    do_task(task, model_id, data.text)
                })
                .fail(
                    function() {
                        console.log('Failed.');
                        $("#resultregion").html('Failed.');
                    });
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

    $.get( "/ppnexamples")
        .done(
            function( data ) {
                var tmp="";
                $.each(data,
                    function(index, item){

                        tmp += '<option value="' + item.ppn + '">' + item.name + '</option>'
                    });
                    $('#ppnexamples').html(tmp);
                }
            );

    task_select()
});
