# Generated by Django 3.1 on 2021-10-02 11:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Home', '0003_image_video_file'),
    ]

    operations = [
        migrations.AddField(
            model_name='image',
            name='image_file',
            field=models.CharField(default='No Location', max_length=500),
        ),
    ]
